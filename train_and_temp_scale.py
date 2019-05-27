import argparse
import random
import sys

from train_and_run_stability_test_bc import get_parser

sys.path.append('/data/rishabh/')
sys.path.append('/Users/apple/MEngProject/')

args, extras = get_parser().parse_known_args()
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
