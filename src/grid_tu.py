import os
import pathlib
import argparse
from datetime import datetime

# script for running grid search on tu datasets

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PTC_MR', choices=["PTC_MR", "MUTAG", "NCI1", "PROTEINS", "IMDB-BINARY","IMDB-MULTI"])
parser.add_argument('--d', type=int, default=1,
                    help='distance of neighbourhood (default: 1)')
parser.add_argument('--t', type=int, default=2,
                    help='size of t-subsets (default: 2)')
parser.add_argument('--single_mlp', action='store_true',
                    help='gnn layer type, allowed are gnn and gmn')
parser.add_argument('--gnn_layer_type', default='gnn',
                    help='gnn layer type, allowed are gnn and gmn')
parser.add_argument('--p_inclusion', type=float, default=1.0,
                    help="probability of inclusion of a subgraph")

args = parser.parse_args()

neighbourhood = [(args.t, args.d)]
hidden = [32, 64, 128]
batch_sizes = [32, 128]
dropouts = [0, 0.5]

for d, t in neighbourhood:
  for emb_dim in hidden:
    for batch_size in batch_sizes:
      for dropout in dropouts:
        cmd = f'python3 tu.py --dataset {args.dataset} --d {d} --t {t} --drop_ratio {dropout} --emb_dim {emb_dim} --batch_size {batch_size} {"--single_mlp" if args.single_mlp else ""} --gnn_layer_type {args.gnn_layer_type} --p_inclusion {args.p_inclusion}'
        print(cmd)

        results_dir = f"results/{args.dataset}/"
        pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True) 
        exp_date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        results_file = f'{results_dir}{cmd} {exp_date}.txt'

        result = os.popen(cmd).read()
        f = open(results_file, "w")
        f.write(result)
        f.close()
        print(result)

