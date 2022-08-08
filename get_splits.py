import numpy as np
from dataset import load_nc_dataset
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10,
                    help='total runs')
parser.add_argument('--train_prop', type=float, default=0.1,
                    help='training set proportion')
parser.add_argument('--valid_prop', type=float, default=0.1,
                    help='validation set proportion')
parser.add_argument('--dataset', type=str, default='cornell')
parser.add_argument('--sub_dataset', type=str, default='')
parser.add_argument('--splits_path', type=str, default='./data/splits/',
                    help='path to save splits')

args = parser.parse_args()

dataset = load_nc_dataset(args.dataset, args.sub_dataset)
split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                    for _ in range(args.runs)]

np.save(os.path.join(args.splits_path, args.dataset+'-splits.npy'), split_idx_lst, allow_pickle=True)
