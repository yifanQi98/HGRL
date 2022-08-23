import argparse
import random
import os
import time

import numpy as np

from losses import *
from transforms import *
from model import HGRL
from graph_learner import ATT_learner

from dataset import load_nc_dataset, load_arxiv_year_dataset
from data_utils import load_fixed_splits
from eval_tools import kmeans_test

from eval.logistic_regression_eval import linear_eval
from torch_geometric.utils import to_undirected
import torch

from get_figure import visualize


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--mm', type=float, default=0.8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--lr_gamma', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--output_size', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--feat_drop', type=float, default=0.2)
parser.add_argument('--edge_drop', type=float, default=0.2)
parser.add_argument('--K', type=int, default=3, help="number of layer in hierarchial_n2n loss")
parser.add_argument('--dataset', type=str, default='cornell')
parser.add_argument('--sub_dataset', type=str, default='')
parser.add_argument('--directed', action='store_true',
                    help='set to not symmetrize adjacency')
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--scale', type=float, default=0.01, 
                    help='factor that scales the loss of CCA')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--eval_every', type=int, default=1)
parser.add_argument('--k', type=int, default=8, help="number of neighbors of knn augmentation")
parser.add_argument('-activation_learner', type=str,
                    default='relu', choices=["relu", "tanh"])
parser.add_argument('--sparse', action='store_true', default=False)
parser.add_argument('--knn_metric', type=str,
                    default='cosine', choices=['cosine', 'minkowski'])
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--rand_split', action='store_true',
                    help='use random splits')
parser.add_argument('--train_prop', type=float, default=0.1,
                    help='training set proportion')
parser.add_argument('--valid_prop', type=float, default=0.1,
                    help='validation set proportion')
parser.add_argument('--save_results', action='store_true',
                    help='save the results')
parser.add_argument('--Init', type=str, default='random', 
                    help='the init method gamma logits')
parser.add_argument('--method', type=str, default='hn2n+CCA')
parser.add_argument('-ta', '--topology_augmentation', type=str, default='learned', 
                    choices=['learned', 'knn', 'init', 'drop'])
parser.add_argument('--task', type=str, default='node_classification',
                    help='node_cluster/node_classification')
parser.add_argument('--save_figure', action='store_true', default=False)



args = parser.parse_args()

seed_it(args.seed)

print(args)

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

### Load and preprocess data ###
dataset = load_nc_dataset(args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

# get the splits for all runs
if args.rand_split or args.dataset == 'ogbn-proteins':
    print("Using random splits")
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
else:
    print("Using fixed splits")
    split_idx_lst = load_fixed_splits(args.dataset, args.sub_dataset)

num_nodes = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
num_classes = max(dataset.label.max().item() + 1, dataset.label.shape[1])
feat_dimension = dataset.graph['node_feat'].shape[1]

# whether or not to symmetrize
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

print(f"num nodes {num_nodes} | num classes {num_classes} | num node feats {feat_dimension}")


def train(model, graph_learner, optimizer, data, args):
    model.train()

    optimizer.zero_grad()

    x1, x2 = data.graph['node_feat'], feat_drop(data.graph['node_feat'], p=args.feat_drop)
    edge_index = data.graph['edge_index']
    
    knn_edge_index = KNN_graph(data.graph['node_feat'], k=args.k)
    drop_edge_index = edge_drop(data.graph['edge_index'], p=args.edge_drop)

    learned_adj = graph_learner(data.graph['node_feat'])
    learned_edge_index = torch.nonzero(learned_adj).t()
    learned_edge_weight = learned_adj[learned_edge_index[0], learned_edge_index[1]]

    h1 = model(x1)
    h2 = model(x2)

    if args.topology_augmentation=='learned':
        # learned adj
        hs1 = model.prop(
            h1, learned_edge_index, learned_edge_weight)
    elif args.topology_augmentation=='knn':
        # knn adj
        hs1 = model.prop(h1, knn_edge_index)
    elif args.topology_augmentation=='drop':
        # drop adj
        hs1 = model.prop(h1, drop_edge_index)
    elif args.topology_augmentation=='init':
        # init adj
        hs1 = model.prop(h1, edge_index)
    else:
        raise ValueError("Unrecognized augmentation")


    alpha = args.alpha
    # hn2n+CCA(h1, h2)
    loss = alpha*model.hierarchial_n2n(h1, hs1) + (1-alpha)*args.scale*CCA_SSG(h1, h2, beta=0)

    print(f"loss: {loss.item()}")

    loss.backward()

    optimizer.step()



def test(model, data, epoch, args, split_idx=None, task='node_classification'):

    model.eval()

    with torch.no_grad():
        representations = model.get_embedding(
            data.graph['node_feat'])
        labels = data.label.squeeze(1).cpu().numpy()
        print(
            f"gamma: {F.softmax(model.logits.detach().cpu(), dim=-1).numpy()}")
        print(
            f"logits: {model.logits.detach().cpu().numpy()}")
    
    if args.save_figure and epoch % 50 == 0:
        visualize(representations, data.label.squeeze(1), args.method, args.dataset, epoch)

    if task == 'node_classification':
        result = linear_eval(
            representations.cpu().numpy(), data.label.squeeze(1).cpu().numpy(), split=split_idx)

        print(f"micro_f1: {result['micro_f1']}")
        print(f"macro_f1: {result['macro_f1']}")
    elif task == 'node_cluster':
        result = kmeans_test(representations, dataset.label.squeeze(1), n_clusters=num_classes, repeat=1)
        
        print(f'Epoch: {epoch:02d}, '
                    f'acc: {100 * result[0]:.2f}%, '
                    f'nmi: {100 * result[2]:.2f}%, '
                    f'ari: {100 * result[4]:.2f}%, ')

    return result


best_results = []
for run in range(args.runs):
    model = HGRL(dataset, args)
    graph_learner = ATT_learner(2, isize=feat_dimension, k=args.k, knn_metric=args.knn_metric,
                                i=6, sparse=args.sparse, mlp_act=args.activation_learner)

    model = model.to(device)
    graph_learner = graph_learner.to(device)

    print(model)

    optimizer = torch.optim.Adam([
        {
            'params': filter(lambda x: x is not model.logits, model.parameters()),
            'lr': args.lr,
            'weight_decay': args.weight_decay
        },
        {
            'params': [model.logits],
            'lr': args.lr_gamma,
            'weight_decay': 0.0
        },
        {
            'params': graph_learner.parameters(),
            'lr': args.lr,
            'weight_decay': args.weight_decay
        }
    ],
        lr=args.lr,
        weight_decay=args.weight_decay)

    split_idx = split_idx_lst[run]
    all_results = []
    for epoch in range(args.epochs):
        start_time = time.time()
        train(model, graph_learner, optimizer, dataset, args)
        end_time = time.time()
        mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
        print(f"run: {run}, epoch: {epoch}, model memory: {mem_params}, epoch time: {end_time-start_time}")
        result = test(model, dataset, epoch, args, split_idx=split_idx, task=args.task)
        all_results.append(result)

    if args.task == 'node_classification':
        best_micro = 0
        best_macro = 0
        for res in all_results:
            if res["micro_f1"] > best_micro:
                best_micro = res["micro_f1"]
                best_macro = res["macro_f1"]
        best_results.append((best_micro, best_macro))
    elif args.task == 'node_cluster':
        result = torch.tensor(all_results)

        best_acc = result[:, 0].max()
        best_nmi = result[:, 2].max()
        best_ari = result[:, 4].max()

        print(f'Highest acc: {100*result[:, 0].max():.2f}')
        print(f'Highest nmi: {100*result[:, 2].max():.2f}')
        print(f'Highest ari: {100*result[:, 4].max():.2f}')

        best_results.append((best_acc, best_nmi, best_ari))
    else:
        raise ValueError("Unrecognized task")

if args.task == 'node_classification':
    best_result = 100*torch.tensor(best_results)
    best_test_micro_f1 = best_result[:, 0]
    best_test_macro_f1 = best_result[:, 1]
    print(f"test micro-f1: {best_test_micro_f1.mean():.3f} ± {best_test_micro_f1.std():.3f}," +
          f"test macro-f1: {best_test_macro_f1.mean():.3f} ± {best_test_macro_f1.std():.3f}, " +
          f"{args.__repr__()}\n")
elif args.task == 'node_cluster':
    best_result = 100*torch.tensor(best_results)
    best_acc = best_result[:, 0]
    best_nmi = best_result[:, 1]
    best_ari = best_result[:, 2]
    print(f"acc: {best_acc.mean():.3f} ± {best_acc.std():.3f}," +
          f"nmi: {best_nmi.mean():.3f} ± {best_nmi.std():.3f}," +
          f"ari: {best_ari.mean():.3f} ± {best_ari.std():.3f}, " +
          f"{args.__repr__()}\n")

prefix = './results'
if args.save_results and args.task == 'node_classification':
    filename = f'{prefix}/{args.task}/{args.method}-{args.dataset}.txt'
    if not os.path.exists(f'{prefix}/{args.task}'):
        os.makedirs(f'{prefix}/{args.task}')
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"test micro-f1: {best_test_micro_f1.mean():.3f} ± {best_test_micro_f1.std():.3f}," +
                        f"test macro-f1: {best_test_macro_f1.mean():.3f} ± {best_test_macro_f1.std():.3f}, " +
                        f"{args.__repr__()}\n")

if args.save_results and args.task == 'node_cluster':
    filename = f'{prefix}/{args.task}/{args.method}-{args.dataset}.txt'
    if not os.path.exists(f'{prefix}/{args.task}'):
        os.makedirs(f'{prefix}/{args.task}')
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"acc: {best_acc.mean():.3f} ± {best_acc.std():.3f}," +
                        f"nmi: {best_nmi.mean():.3f} ± {best_nmi.std():.3f}," +
                        f"ari: {best_ari.mean():.3f} ± {best_ari.std():.3f}, " +
                        f"{args.__repr__()}\n")
