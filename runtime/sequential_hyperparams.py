#!/usr/bin/python
import numpy as np
import itertools
import os
import subprocess
import random
import argparse

def signif(x, p):
    #https://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy/59888924#59888924
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

def make_hyperparams(start_lr: int = -4,
                     end_lr: int = -2,
                     n_lrs: int = 10,
                     end_wd: float = 2.,
                     n_wds: int = 10,
                     n_channels: int = 2,
                     n_degrees: int = 7,
                     end_layers: int = 20,
                     n_choices: int = 50):
    lrs = signif(np.logspace(start_lr, end_lr, n_lrs), 3)
    weight_decays = signif(np.geomspace(0.000001, end_wd, n_wds), 3)
    layers = [layer + 1 for layer in range(end_layers)]
    channels = [channel + 2 for channel in range(n_channels)]
    degrees = [int(2**x) for x in np.arange(3, n_degrees)]
    options = list(itertools.product(lrs, weight_decays, layers, channels, degrees))
    return(random.sample(options, n_choices))

def run_training(hyp_params):
    subprocess.check_call(['/home/sabari/ProteinSol/Topoformer/scripts/train_multi_gpu_proteins_barcodes.sh', '40', 'false', '130', *[str(x) for x in hyp_params]])

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("start_lr", type = int, default = -4)
    parser.add_argument("end_lr", type = int, default = -2)
    parser.add_argument("n_lrs", type = int, default = 10)
    parser.add_argument("end_wd", type = float, default = 2)
    parser.add_argument("n_wds", type = int, default = 10)
    parser.add_argument("n_channels", type = int, default = 2)
    parser.add_argument("n_degrees", type = int, default = 7)
    parser.add_argument("end_layers", type = int, default = 20)
    parser.add_argument('n_choices', type = int, default = 50)
    args=parser.parse_args()
    return args

if __name__ == '__main__':
    inputs = parse_args()
    hyperparams = make_hyperparams(inputs.start_lr,
                                   inputs.end_lr,
                                   inputs.n_lrs,
                                   inputs.end_wd,
                                   inputs.n_wds,
                                   inputs.n_channels,
                                   inputs.n_degrees,
                                   inputs.end_layers,
                                   inputs.n_choices)
    for param in hyperparams:
        run_training(param)
