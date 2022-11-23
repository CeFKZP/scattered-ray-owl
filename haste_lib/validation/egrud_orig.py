from enum import Enum
from typing import Union, Optional, Tuple
import math

import torch
import torch.jit
from torch import Tensor
from torch import nn as nn
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence

from haste_pytorch.egru import SpikeFunction


class EVNNThresholdInit(Enum):
    zero_scalar = 'zero-scalar'
    zero_vector = 'zero-vector'
    rand_vector = 'random-vector'
    const_scalar = 'const-scalar'


class ScriptEGRUD(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size: int,
                 thr_init: EVNNThresholdInit,
                 init_like_lstm: bool = False,
                 thr_init_scale: float = 1,
                 dampening_factor: float = 0.7,
                 use_exponential_pseudo_derivative=False,
                 layer_norm: bool = False,
                 dropout_connect: float = 0.0,
                 use_output_trace=False,
                 pseudo_derivative_width=1.,
                 ):
        super().__init__()
        print("Using ScriptEGRUD")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = 1
        self.use_output_trace = use_output_trace
        self.alpha = torch.tensor(0.9)

        # update gate
        self.W = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.U = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.b_w = Parameter(torch.Tensor(3 * hidden_size))
        self.b_u = Parameter(torch.Tensor(3 * hidden_size))

        self.init_weights(init_like_lstm)

        self.gru = ScriptGRU(input_size=input_size, hidden_size=hidden_size, layer_norm=layer_norm,
                             U=self.U, W=self.W, bias_u=self.b_u, bias_w=self.b_w)

        self.dampening_factor = Parameter(torch.Tensor([dampening_factor]), requires_grad=False)
        if use_exponential_pseudo_derivative:  # True == 1
            self.use_exponential_pseudo_derivative = Parameter(torch.Tensor([1]), requires_grad=False)
        else:  # False == 0 (Verified that it works with torch.torch.Tensor also)
            self.use_exponential_pseudo_derivative = Parameter(torch.Tensor([0]), requires_grad=False)
        self.pseudo_derivative_width = Parameter(torch.Tensor([pseudo_derivative_width]), requires_grad=False)

        # If the threshold is positive, reset is more meaningful
        if thr_init == EVNNThresholdInit.const_scalar:
            self.thr_reparam = Parameter(thr_init_scale * torch.Tensor([1.]))

        elif thr_init == EVNNThresholdInit.rand_vector:
            self.thr_reparam = Parameter(thr_init_scale * torch.normal(torch.zeros(self.hidden_size),
                                                                       math.sqrt(2) * torch.ones(self.hidden_size)))
        else:
            raise RuntimeError(f"Unsupported threshold initialization {thr_init}")

        self.thr = torch.sigmoid(self.thr_reparam)

        # for cell state regularization
        # self.cell_states = []
        # self.output_gates = []

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.W.device),
                # hack to access device of the model
                torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.W.device),
                torch.zeros(self.n_layers, batch_size, 3 * self.hidden_size).to(self.W.device))

    def init_weights(self, like_lstm=False):
        for k, v in self.named_parameters():
            if k in ['thr_reparam', 'alpha', 'alpham1', 'dampening_factor', 'use_exponential_pseudo_derivative', 'pseudo_derivative_width']:
                continue
            elif like_lstm:
                stdv = 1.0 / math.sqrt(self.hidden_size)
                nn.init.uniform_(v, -stdv, stdv)
            elif v.data.ndimension() >= 2:
                nn.init.xavier_uniform_(v)
            else:
                nn.init.uniform_(v)

    def forward(self,
                seq: Union[torch.Tensor, PackedSequence],
                init_states: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        # is_packed = isinstance(seq, PackedSequence)
        # if is_packed:
        #     # seq, batch_sizes, sorted_indices, unsorted_indices = seq
        #     # max_batch_size = int(batch_sizes[0])
        #     seq_unpacked, lens_unpacked = pad_packed_sequence(seq, batch_first=True)
        #     x = seq_unpacked
        # else:
        x = seq

        batch_size, seq_len, _ = x.size()

        if init_states is None:
            c_tm_layer, o_tm_layer, i_tm_layer, tr_tm_layer = \
                torch.zeros(self.n_layers, batch_size, self.hidden_size).to(x.device), \
                torch.zeros(self.n_layers, batch_size, self.hidden_size).to(x.device), \
                torch.zeros(self.n_layers, batch_size, 3 * self.hidden_size).to(x.device), \
                torch.zeros(self.n_layers, batch_size, self.hidden_size).to(x.device)
        else:
            c_tm_layer, o_tm_layer, i_tm_layer, tr_tm_layer = init_states

        c_t = torch.squeeze(c_tm_layer, 0)
        o_t = torch.squeeze(o_tm_layer, 0)
        i_t = torch.squeeze(i_tm_layer, 0)
        tr_t = torch.squeeze(tr_tm_layer, 0)

        # self.cell_states = [c_t]
        # self.output_gates = [o_t]

        self.thr = torch.sigmoid(self.thr_reparam.to(x.device))

        h_t, c_t = output_gate(c_t, o_t, self.thr)

        # for now only support single layer EGRU
        assert self.n_layers == 1

        hidden_states = torch.empty((seq_len, batch_size, self.hidden_size), device=x.device)
        c = torch.empty((seq_len, batch_size, self.hidden_size), device=x.device)
        o = torch.empty((seq_len, batch_size, self.hidden_size), device=x.device)
        i = torch.empty((seq_len, batch_size, 3 * self.hidden_size), device=x.device)
        tr = torch.empty((seq_len, batch_size, self.hidden_size), device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]

            c_t, i_t = self.gru(x_t, (h_t, c_t, i_t))

            o_t = SpikeFunction.apply(c_t - self.thr, self.dampening_factor, self.pseudo_derivative_width)

            # record values for activity regularization
            # self.cell_states.append(c_t)
            # self.output_gates.append(o_t)

            # reset the gate on all but the last step
            if t < seq_len - 1:
                h_t, c_t = output_gate(c_t, o_t, self.thr)
            else:
                h_t = hadamard(o_t, c_t)

            if self.use_output_trace:
                # tr_t = self.alpha * tr_t + (1 - self.alpha) * h_t
                tr_t = trace(tr_t, h_t, self.alpha)

            # record outputs
            hidden_states[t] = h_t
            c[t] = c_t
            o[t] = o_t
            i[t] = i_t
            tr[t] = tr_t

        # if is_packed:
        #     hidden_states = pack_padded_sequence(hidden_states, lens_unpacked, batch_first=True)

        c_t_layer = torch.unsqueeze(c, 0)
        o_t_layer = torch.unsqueeze(o, 0)
        i_t_layer = torch.unsqueeze(i, 0)
        tr_t_layer = torch.unsqueeze(tr, 0)

        hidden_states = torch.transpose(hidden_states, 0, 1)

        return hidden_states, (c_t_layer, o_t_layer, i_t_layer, tr_t_layer)


class ScriptGRU(nn.Module):
    def __init__(self, input_size, hidden_size, layer_norm, U, W, bias_u, bias_w):
        super(ScriptGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = 1

        self.W = W
        self.U = U
        self.b_u = bias_u
        self.b_w = bias_w

        # layer normalization, should not be initialized
        self.layernorm_ih = nn.LayerNorm(3 * hidden_size) if layer_norm else nn.Identity()
        self.layernorm_hh = nn.LayerNorm(3 * hidden_size) if layer_norm else nn.Identity()
        self.layernorm_gate = nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()

        # tau_syn_ms = 0.14476482730108395
        # self.alpha = Parameter(torch.exp(torch.Tensor([-1/tau_syn_ms])), requires_grad=False)
        # self.alpham1 = Parameter(torch.expm1(torch.Tensor([-1/tau_syn_ms])), requires_grad=False)
        self.alpha = Parameter(torch.Tensor([0.000]),
                               requires_grad=False)
        self.alpham1 = Parameter(torch.Tensor([self.alpha - 1]),
                                 requires_grad=False)

    def init_weights(self, like_lstm=False):
        for k, v in self.named_parameters():
            if k in ['thr_reparam', 'alpha', 'alpham1', 'dampening_factor', 'use_exponential_pseudo_derivative']:
                continue
            elif like_lstm:
                stdv = 1.0 / math.sqrt(self.hidden_size)
                nn.init.uniform_(v, -stdv, stdv)
            elif v.data.ndimension() >= 2:
                nn.init.xavier_uniform_(v)
            else:
                nn.init.uniform_(v)

    # @torch.jit.script_method
    def forward(self, x, state):
        # type: (Tensor, Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]
        hidden, cell_state, I = state
        x_results = self.layernorm_ih(torch.matmul(x, self.W.t()) + self.b_w)
        h_results = self.layernorm_hh(torch.matmul(hidden, self.U.t()) + self.b_u)

        i_u, i_r, i_c = torch.tensor_split(I, 3, dim=-1)
        x_u, x_r, x_c = x_results.chunk(3, 1)
        h_u, h_r, h_c = h_results.chunk(3, 1)

        i_r = torch.mul(self.alpha, i_r) - torch.mul(self.alpham1, x_r + h_r)
        r = torch.sigmoid(i_r)

        i_u = torch.mul(self.alpha, i_u) - torch.mul(self.alpham1, x_u + h_u)
        u = torch.sigmoid(i_u)

        i_c = torch.mul(self.alpha, i_c) - torch.mul(self.alpham1, x_c + torch.mul(r, h_c))
        z = torch.tanh(i_c)

        I = torch.cat((i_u, i_r, i_c), dim=-1)

        return self.layernorm_gate(torch.mul(u, cell_state) + torch.mul(1 - u, z)), I


@torch.jit.script
def trace(tr, h, alpha):
    return alpha * tr + (1 - alpha) * h

@torch.jit.script
def full_trace(output, alpha):
    """
    Assume batch-first is True
    """

    tr_vals = torch.zeros_like(output)
    for t in range(1, output.shape[1]):
        # tr_vals[:, t, :] = alpha * tr_vals[:, t - 1, :] + (1 - alpha) * output[:, t, :]
        tr_vals[:, t, :] = trace(tr_vals[:, t - 1, :], output[:, t, :], alpha)

    return tr_vals



@torch.jit.script
def output_gate(c, o, threshold):
    h = torch.mul(c, o)
    c_reset = c - torch.mul(o, threshold)
    return h, c_reset


@torch.jit.script
def hadamard(x, y):
    return x * y


if __name__ == '__main__':
    from evnn.egrud import EGRUD as EGRUD_original

    bs = 16
    seq_len = 35

    ninp = 32
    nout = 32
    nlayers = 1

    # inputs in range (-0.1, 0.1)
    inputs = torch.rand(bs, seq_len, ninp) * 0.2 - 0.1

    egrud = EGRUD_original(input_size=ninp, n_units=nout, thr_init=EVNNThresholdInit.const_scalar)
    egrud_script = ScriptEGRUD(input_size=ninp, hidden_size=nout, thr_init=EVNNThresholdInit.const_scalar)

    # same parameters
    egrud_script.thr_reparam = Parameter(egrud.thr)
    egrud_script.W = Parameter(torch.cat([egrud.W_u.t(), egrud.W_r.t(), egrud.W_c.t()]))
    egrud_script.U = Parameter(torch.cat([egrud.U_u.t(), egrud.U_r.t(), egrud.U_c.t()]))
    egrud_script.b_u = Parameter(torch.cat([egrud.b_u, egrud.b_r, egrud.b_c]))
    egrud_script.b_w = Parameter(torch.cat([egrud.a_u, egrud.a_r, egrud.a_c]))

    states, states_script = None, None

    outputs = []
    outputs_script_loop = []
    with torch.no_grad():
        for i in range(seq_len):
            x = inputs[:, i, :].unsqueeze(1)
            hidden, states = egrud(x, states)
            outputs.append(hidden)

            hidden_script, states_script = egrud_script(x, states_script)
            outputs_script_loop.append(hidden_script)

    outputs = torch.cat(outputs, dim=1)
    outputs_script_loop = torch.cat(outputs_script_loop, dim=1)
    outputs_script, _ = egrud_script(inputs, None)

    print("Baseline vs Loop Script", torch.allclose(outputs, outputs_script_loop))
    print("Baseline vs Joint Script", torch.allclose(outputs, outputs_script))
