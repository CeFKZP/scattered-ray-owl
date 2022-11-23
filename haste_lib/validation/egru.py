# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
from time import time
from unittest import mock
import torch
import haste_pytorch as haste
from egrud_orig import ScriptEGRUD, EVNNThresholdInit

import numpy as np
import pandas as pd

seed = 5595
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

RNN_MAP = {
    'egru': haste.EGRU,
    'gru': haste.GRU,
}

HASTE_TO_NATIVE = {
    haste.GRU: torch.nn.GRU,
    haste.LSTM: torch.nn.LSTM,
}

batch_size = 12
time_steps = 16
input_size = 2
hidden_size = 4


def self_consistency(rnn, x, seed=5566):
  x_cuda = x.clone().cuda()
  x_cuda_torch = x_cuda.detach().clone()
  x_cuda.requires_grad_(True)
  x_cuda_torch.requires_grad_(True)

  rnn.cuda()
  y1, (h1, o1, t1) = rnn.forward(x_cuda)
  t1.backward(torch.ones_like(t1), retain_graph=True)
  y1.backward(torch.ones_like(y1), retain_graph=True)
  h1.backward(torch.ones_like(h1), retain_graph=True)
  o1.backward(torch.ones_like(o1), retain_graph=False)

  torch.manual_seed(seed)
  with mock.patch.object(rnn, "use_custom_cuda", False):
    y2, (h2, o2, t2) = rnn.forward(x_cuda_torch)
  y2.backward(torch.ones_like(y2), retain_graph=True)
  t2.backward(torch.ones_like(t2), retain_graph=True)
  h2.backward(torch.ones_like(h2), retain_graph=True)
  o2.backward(torch.ones_like(o2), retain_graph=False)

  g2 = x_cuda_torch.grad.data
  g1 = x_cuda.grad.data

  print("forward error  y {:.8f}".format(torch.max(torch.abs(y1-y2).cpu())))
  print("forward error  h {:.8f}".format(torch.max(torch.abs(h1-h2).cpu())))
  print("forward error  o {:.8f}".format(torch.max(torch.abs(o1-o2).cpu())))
  print("trace error {:.6f}".format(torch.max(torch.abs(t1-t2).cpu())))
  
  print("backward error x {:.8f}".format(torch.max(torch.abs(g1-g2).cpu())))

def native_consistency(haste_rnn, pytorch_rnn, x):
  pytorch_rnn.cuda()
  haste_rnn.cuda()
  haste_rnn.from_native_weights(
      pytorch_rnn.W,
      pytorch_rnn.U,
      pytorch_rnn.b_w,
      pytorch_rnn.b_u,
      pytorch_rnn.thr_reparam)

  x1 = x.clone().cuda()
  x2 = x.clone().cuda()
  x1.requires_grad_(True)
  x2.requires_grad_(True)

  haste_rnn.use_custom_cuda = False
  y1, (h1, o1, t1) = haste_rnn.forward(x1)
  y1.backward(torch.ones_like(y1))

  y2, (h2, o2, _, t2) = pytorch_rnn.forward(x2)
  y2.backward(torch.ones_like(y2))

  g1 = x1.grad.data
  g2 = x2.grad.data

  print("native to haste forward error  y {:.8f}".format(torch.max(torch.abs(y1-y2).cpu())))
  print("native to haste backward error x {:.8f}".format(torch.max(torch.abs(g1-g2).cpu())))


def time_measure(rnn, x, num_trials=10):
  x_cuda = x.clone().cuda()
  x_cuda_torch = x_cuda.detach().clone()
  x_cuda.requires_grad_(True)
  x_cuda_torch.requires_grad_(True)

  rnn.cuda()
  cpp_time = 0
  pytorch_time = 0
  cpp_bwd_time = 0
  pytorch_bwd_time = 0

  cpp_oom_flag = False
  pytorch_oom_flag = False
  for trial in range(num_trials):
    seed = 5566 + trial
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    try:
      start = time()
      y1, _ = rnn.forward(x_cuda)
      cpp_time += time() - start
      start = time()
      y1.backward(torch.ones_like(y1))
      cpp_bwd_time += time() - start
    except RuntimeError as e:
      print('OOM while calculating cpp')
      cpp_oom_flag = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    with mock.patch.object(rnn, "_is_cuda", lambda: False):
      try:
        start = time()
        y2, _ = rnn.forward(x_cuda_torch)
        pytorch_time += time() - start

        start = time()
        y2.backward(torch.ones_like(y2))
        pytorch_bwd_time += time() - start
      except RuntimeError as e:
        print('OOM while calculating pytorch')
        pytorch_oom_flag = True

  result = {'cpp': {'fwd': float('nan') if cpp_oom_flag else cpp_time/num_trials,
                    'bwd': float('nan') if cpp_oom_flag else cpp_bwd_time/num_trials},
            'pytorch': {'fwd': float('nan') if pytorch_oom_flag else pytorch_time/num_trials,
                        'bwd': float('nan') if pytorch_oom_flag else pytorch_bwd_time/num_trials}}
  return result

def loop_timing_measurement(rnn):
    df = pd.DataFrame(columns=['batch', 'time_steps', 'input_size', 'hidden_size', 'cpp_fwd', 'pytorch_fwd', 'cpp_bwd', 'pytorch_bwd'])
    for b in np.exp2(np.arange(4, 9)):
      for t in np.exp2(np.arange(4, 6)):
        for in_size in np.exp2(np.arange(5, 15)):
          for h_size in np.exp2(np.arange(6, 15)):
            print(str(b) + ' ' + str(t) + ' ' + str(in_size) + ' ' + str(h_size))
            try:
              x = torch.rand(int(b), int(t), int(in_size), dtype=torch.float32)
              egru = rnn(int(in_size), int(h_size), batch_first=True)
            except RuntimeError as e:
              print('OOM while creating model')
              continue
            result = time_measure(egru, x)
            print("forward - cpp: {:.4f}ms, pytorch:{:.4f}ms".format(result['cpp']['fwd']*1000/t, result['pytorch']['fwd']*1000/t))
            print("backward - cpp: {:.4f}ms, pytorch:{:.4f}ms".format(result['cpp']['bwd']*1000/t, result['pytorch']['bwd']*1000/t))
            df = df.append({'batch' : b,
            'time_steps' : t, 'input_size' : in_size, 'hidden_size' : h_size,
            'cpp_fwd' : result['cpp']['fwd']/t, 'pytorch_fwd': result['pytorch']['fwd']/t, 'cpp_bwd' : result['cpp']['bwd']/t, 'pytorch_bwd': result['pytorch']['bwd']/t},
            ignore_index=True)

    # df.to_csv('egru_forward_timings.csv')

def main():
  rnn = RNN_MAP['egru']

  x = torch.rand(batch_size,time_steps, input_size, dtype=torch.float32)
  egru = rnn(input_size, hidden_size, zoneout=0.0, batch_first=True)
  self_consistency(egru, x, seed=seed)

  native_egru = ScriptEGRUD(input_size, hidden_size, thr_init_scale=1., thr_init=EVNNThresholdInit.rand_vector)
  native_consistency(egru, native_egru,x)
  
  # loop_timing_measurement(rnn)
  return



if __name__ == '__main__':
  main()
