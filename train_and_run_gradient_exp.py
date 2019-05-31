import argparse
import random
import sys

from train_and_run_stability_test_bc import get_parser

sys.path.append('/data/rishabh/')
sys.path.append('/Users/apple/MEngProject/')
sys.path.append('/home/ubuntu/')

if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument("--grad", type=int, default=0)

    args, extras = get_parser().parse_known_args()
    args.extras = extras

    from Transparency.Trainers.DatasetBC import *
    from Transparency.ExperimentsBC import *
    from common_code.common import pickle_to_file
    from torch.backends import cudnn

    dataset = datasets[args.dataset](args)

    if args.output_dir is not None:
        dataset.output_dir = args.output_dir

    encoders = ['cnn', 'lstm', 'average'] if args.encoder == 'all' else [
        args.encoder]

    seeds = eval(args.seeds)

    all_grad_outputs = []
    for pseudo_random_seed in seeds:
        os.environ['PYTHONHASHSEED'] = str(pseudo_random_seed)
        np.random.seed(pseudo_random_seed)
        random.seed(pseudo_random_seed)
        torch.manual_seed(pseudo_random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        grads = []
        if args.loss:
            train_losses = []

        if args.attention in ['tanh', 'all']:
            grads = train_dataset_and_get_gradient(
                dataset, encoders, args.iters)

        if args.attention in ['dot', 'all']:
            encoders_temp = [e + '_dot' for e in encoders]
            grads = train_dataset_and_get_gradient(
                dataset, encoders_temp, args.iters)

        all_grad_outputs.append([grads])

    run_settings_str = args.name + args.swa + args.seeds + str(
        args.attention) + str(args.dataset) + str(args.encoder) + str(args.temp)
    file_name = "gradient-outputs-" + run_settings_str + ".pkl"
    pickle_to_file(all_grad_outputs, file_name)