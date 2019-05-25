import argparse
import numpy as np
import os
import random
import torch
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
parser.add_argument('--path', type=str, required=True)

args, extras = parser.parse_known_args()
args.extras = extras

pseudo_random_seed = 21
os.environ['PYTHONHASHSEED'] = str(pseudo_random_seed)
np.random.seed(pseudo_random_seed)
random.seed(pseudo_random_seed)
torch.manual_seed(pseudo_random_seed)

from Transparency.Trainers.DatasetBC import *
from Transparency.ExperimentsBC import *

dataset = datasets[args.dataset](args)

if args.output_dir is not None:
    dataset.output_dir = args.output_dir

encoders = ['cnn', 'lstm', 'average'] if args.encoder == 'all' else [
    args.encoder]

if args.attention in ['tanh', 'all']:
    # generate_graphs_on_encoders(dataset, encoders)
    eval_swa_model(dataset, args.path)
if args.attention in ['dot', 'all']:
    encoders = [e + '_dot' for e in encoders]
    eval_swa_model(dataset, args.path)
    # generate_graphs_on_encoders(dataset, encoders)