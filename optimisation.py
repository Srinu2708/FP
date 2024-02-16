import argparse

import yaml

import wandb
from bayes.bayes_search import bayes_search_next_runs
from train import main

parser = argparse.ArgumentParser(description='Segmentation pipeline')

parser.add_argument('--model_name', default=None, type=str, help='model name')
parser.add_argument('--project', default="sweeps", type=str, help='project name')
parser.add_argument('--root', default=None, type=str, help='root to the optimisation file')
parser.add_argument('--file', default=None, type=str, help='optimisation file')
parser.add_argument('--sweep_id', default=None, type=str, help='sweep id')
parser.add_argument('--count', default=1, type=int, help='number of training')
parser.add_argument('--use_local_controller', default=True, type=bool, help='use classic or local controller')
parser.add_argument('--use_customized_bayes', default=True, type=bool,
                    help='use our bayes search implementation or not')

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.root + args.file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config['parameters']['model_name'] = {'values': [args.model_name]}
        config['parameters']['config_file'] = {'values': [args.root + args.model_name + '_config.yaml']}
    sweep_id = wandb.sweep(config, project=args.project)

    if args.sweep_id is not None:
        sweep_id = args.sweep_id

    if args.use_local_controller:
        local_controller = wandb.controller(sweep_id)
        if args.use_customized_bayes:
            local_controller.configure_search(bayes_search_next_runs)
        for i in range(args.count):
            local_controller.step()
            wandb.agent(sweep_id, main, count=1)

    else:
        wandb.agent(sweep_id, main, count=args.count)
