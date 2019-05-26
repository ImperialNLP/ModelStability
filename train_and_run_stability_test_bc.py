import argparse
import numpy as np
import os
import random
import torch
import pickle
import sys

sys.path.append('/data/rishabh/')

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
parser.add_argument('--seeds', nargs='?', default='[0,50,25,0.001]',
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

all_outputs = []
for pseudo_random_seed in seeds:
    os.environ['PYTHONHASHSEED'] = str(pseudo_random_seed)
    np.random.seed(pseudo_random_seed)
    random.seed(pseudo_random_seed)
    torch.manual_seed(pseudo_random_seed)
    preds, atns = [], []
    if args.attention in ['tanh', 'all']:
        preds, atns = train_dataset_and_get_atn_map(dataset, encoders)
        # generate_graphs_on_encoders(dataset, encoders)
    if args.attention in ['dot', 'all']:
        encoders = [e + '_dot' for e in encoders]
        preds, atns = train_dataset_and_get_atn_map(dataset, encoders)
        # generate_graphs_on_encoders(dataset, encoders)
    all_outputs.append((preds, atns))

swa_settings = eval(args.swa)
if swa_settings[0]:
    file_name = "stability-outputs-swa-" + args.seeds + str(args.attention) + str(
        args.dataset) + str(args.encoder) + ".pkl"
else:
    file_name = "stability-outputs-og-" + args.seeds + str(args.attention) + str(
        args.dataset) + str(args.encoder) + ".pkl"

pkl_file = open(file_name, 'wb')
pickle.dump(all_outputs, pkl_file)
pkl_file.close()
