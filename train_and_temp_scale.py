import argparse
import numpy as np
import os
import random
import torch
import pickle
import sys

sys.path.append('/data/rishabh/')
sys.path.append('/Users/apple/MEngProject/')

parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str)
parser.add_argument('--encoder', type=str,
                    choices=['cnn', 'lstm', 'average', 'all'], required=True)
parser.add_argument('--attention', type=str, choices=['tanh', 'dot', 'all'],
                    required=True)
parser.add_argument('--seeds', nargs='?', default='[2,9001,2**18]',
                    help='Seeds for runs.')
parser.add_argument('--swa', nargs='?', default='[0,50,25,0.001]',
                    help='Enable Stochastic Weighted Averaging (active, start_val, freq, learning-rate).')

args, extras = parser.parse_known_args()
args.extras = extras

from Transparency.Trainers.DatasetBC import *
from Transparency.ExperimentsBC import *

dataset = datasets[args.dataset](args)

if args.output_dir is not None:
    dataset.output_dir = args.output_dir

encoders = ['cnn', 'lstm', 'average'] if args.encoder == 'all' else [
    args.encoder]

seeds = eval(args.seeds)

for pseudo_random_seed in seeds:
    os.environ['PYTHONHASHSEED'] = str(pseudo_random_seed)
    np.random.seed(pseudo_random_seed)
    random.seed(pseudo_random_seed)
    torch.manual_seed(pseudo_random_seed)
    preds, atns = [], []
    if args.attention in ['tanh', 'all']:
        train_dataset_and_temp_scale(dataset, encoders)
    if args.attention in ['dot', 'all']:
        encoders = [e + '_dot' for e in encoders]
        train_dataset_and_temp_scale(dataset, encoders)
