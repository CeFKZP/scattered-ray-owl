import argparse

import numpy as np
import torch
from torch import nn

import lm.data as d
from lm.models import LanguageModel


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=16816)
    argparser.add_argument('--data', type=str, required=False, help='path to datasets')
    argparser.add_argument('--dataset', type=str, default='PTB', choices=['WT2', 'PTB', 'WT103'])
    argparser.add_argument('--scratch', type=str, default='./scratch', help='scratch directory for jobs')
    argparser.add_argument('--batch_size', type=int, default=80)
    argparser.add_argument('--bptt', type=int, default=70)
    argparser.add_argument('--hidden_dim', type=int, default=1350)
    argparser.add_argument('--layers', type=int, default=3)
    argparser.add_argument('--emb_dim', type=int, default=400)
    argparser.add_argument('--dropout_emb', type=float, default=0.0)
    argparser.add_argument('--dropout_words', type=float, default=0.0)
    argparser.add_argument('--dropout_forward', type=float, default=0.0)
    argparser.add_argument('--pseudo_derivative_width', type=float, default=1.0)
    argparser.add_argument('--thr_init_mean', type=float, default=-2.0)

    return argparser.parse_args()


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load dataset
    train_data, val_data, test_data, vocab_size = d.get_data(root=args.data,
                                                             dset=args.dataset,
                                                             batch_size=args.batch_size,
                                                             device=device)

    print(f"Dataset {args.dataset} has {vocab_size} tokens")

    # load the model
    model_args = {
        'nlayers'          : args.layers,
        'projection'       : False,
        'alpha'            : 0,
        'beta'             : 0,
        'gamma'            : 0,
        'emb_dim'          : args.emb_dim,
        'hidden_dim'       : args.hidden_dim,
        'vocab_size'       : vocab_size,
        'dropout_words'    : args.dropout_words,
        'dropout_embedding': args.dropout_emb,
        'dropout_connect'  : 0.0,
        'dropout_forward'  : args.dropout_forward,
    }

    model_native = LanguageModel(**model_args,
                                 rnn_type='egrud',
                                 dampening_factor=0.7,
                                 pseudo_derivative_width=args.pseudo_derivative_width).to(device)

    model_haste = LanguageModel(**model_args,
                                rnn_type='egrud_haste',
                                dampening_factor=0.7,
                                pseudo_derivative_support=args.pseudo_derivative_width,
                                thr_mean=args.thr_init_mean).to(device)

    model_haste.embeddings = model_native.embeddings
    model_haste.decoder = model_native.decoder

    for pytorch_rnn, haste_rnn in zip(model_native.rnns, model_haste.rnns):
        haste_rnn.from_native_weights(
                pytorch_rnn.W,
                pytorch_rnn.U,
                pytorch_rnn.b_w,
                pytorch_rnn.b_u,
                pytorch_rnn.thr_reparam)

    # setup training
    criterion = nn.CrossEntropyLoss()
    optimizer_native = torch.optim.Adam(model_native.parameters())
    optimizer_haste = torch.optim.Adam(model_haste.parameters())

    out_native = step(model=model_native,
                      train_data=train_data,
                      optimizer=optimizer_native,
                      criterion=criterion,
                      batch_size=args.batch_size,
                      bptt=args.bptt,
                      ntokens=vocab_size)
    grad_native = get_grad(model_native)
    model_native.zero_grad()
    model_haste.zero_grad()
    out_haste = step(model=model_haste,
                     train_data=train_data,
                     optimizer=optimizer_haste,
                     criterion=criterion,
                     batch_size=args.batch_size,
                     bptt=args.bptt,
                     ntokens=vocab_size)
    grad_haste = get_grad(model_haste)

    print(f"Mismatch Gates: {torch.logical_xor(out_haste[0] == 0, out_native[0] == 0).float().mean()}")
    print(f"Forward diff:   {torch.max(torch.abs(out_haste[0] - out_native[0]).cpu()):.8f}")
    print(f"Backward diff:  {torch.max(torch.abs(grad_haste - grad_native).cpu()):.8f}")


def get_grad(model):
    grads = []
    for n, param in model.named_parameters():
        if param.requires_grad and 'raw' not in n:
            grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    return grads


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is not None:
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


def step(model,
         train_data,
         optimizer,
         criterion,
         batch_size,
         bptt,
         ntokens, ):
    model.train()  # turn on train mode
    hidden = model.init_hidden(batch_size)

    batch, i = 0, 0

    seq_len = bptt

    # fetch data and make it batch first
    data, targets = d.get_batch(train_data, i, seq_len=seq_len, batch_first=True)

    # prepare forward pass
    hidden = repackage_hidden(hidden)
    optimizer.zero_grad()

    # forward pass
    output, hidden, hid_full, hid_dropped = model(data, hidden)
    loss = criterion(output.view(-1, ntokens), targets)

    print(loss.item())
    # backward pass and weight update
    loss.backward()

    return hid_full


if __name__ == "__main__":
    args = get_args()
    main(args)
