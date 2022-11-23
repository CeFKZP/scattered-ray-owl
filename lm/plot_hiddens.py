import os.path
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse

import torch


def draw_timeseries(hiddens):
    fig, axes = plt.subplots(len(hiddens), 1, figsize=(10, 20), sharex=True)
    for i, hid in enumerate(hiddens):
        # prepare states
        print(hid.shape)
        states = hid[0, :400, 100:300]

        # plot states time series
        art = axes[i].imshow(np.transpose(states), cmap='Greens')

        # annotate plot
        axes[i].set_title(f'Layer {i}')
        axes[i].set_ylabel('Units')

        if i == len(hiddens) - 1:
            axes[i].set_xlabel('t')

    fig.colorbar(art, ax=axes, shrink=0.4, location='bottom')
    fig.savefig(os.path.join(args.file.replace('.hdf', '.png')))


def draw_histogram(hiddens):
    fig, axes = plt.subplots(1, len(hiddens), figsize=(20, 5))

    for i, hid in enumerate(hiddens):
        hid_dim = hid.shape[2]
        hid_cells = hid.reshape(-1, hid_dim)
        seq_len = hid_cells.shape[0]
        spike_frequency = np.sum(hid_cells != 0, axis=0) / seq_len

        # verbose
        print(f"less than 1/100: {np.sum(spike_frequency < 0.01)} / {spike_frequency.shape} // never: {np.sum(hid_cells.sum(axis=0) == 0)} / {spike_frequency.shape}")
        print(f"Layer {i+1} Percentile: {np.round(np.percentile(np.sort(spike_frequency), [0, 1, 5, 10, 20, 50]), 3)}")

        # draw histogram
        axes[i].hist(spike_frequency, bins=50)
        axes[i].set_title(f'Layer {i}')
        #axes[i].set_yscale('log')

    fig.suptitle('Event Frequency by Layer', fontsize=40)
    plt.tight_layout()
    fig.savefig(os.path.join(args.file.replace('.hdf', '_histogram.svg')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True)
    parser.add_argument('-w', '--width', type=float, default=1.0)
    args = parser.parse_args()

    width = np.abs(args.width)

    with h5py.File(args.file, 'r') as f:
        hiddens = [f[hid][:] for hid in list(f) if 'hidden_states' in hid]
        centered_cell_states = [f[hid][:] for hid in list(f) if 'centered_cell_states' in hid]

    draw_timeseries(hiddens)
    print([(ccs.min(), ccs.max()) for ccs in centered_cell_states])
    backward_sparsity = [np.mean(np.logical_or(ccs > 1 / width, ccs < - 1 / width)) for ccs in centered_cell_states]
    print("Backward sparsity:        ", backward_sparsity)
    print("Centered cell state means:", [np.mean(ccs) for ccs in centered_cell_states])
    print("Centered cell state std:  ", [np.std(ccs.astype(np.float32)) for ccs in centered_cell_states])
    print("Dead cells:               ", [np.sum(np.count_nonzero(hid.reshape(-1, hid.shape[-1]), axis=0) == 0) for hid in hiddens])

    draw_histogram(hiddens)
