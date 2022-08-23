import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
import copy

def KNN_graph(x, k=12):
    # KNN-graph
    h = F.normalize(x, dim=-1)
    device = x.device
    logits = torch.matmul(h, h.t())
    _, indices = torch.topk(logits, k=k, dim=-1)
    graph = torch.zeros(h.shape[0], h.shape[0], dtype=torch.int64, device=device).scatter_(1, indices, 1)
    
    edge_index = torch.nonzero(graph).t()
    edge_index = to_undirected(edge_index)
    
    return edge_index

def edge_drop(edge_index, p=0.4):
    # copy edge_index
    edge_index = copy.deepcopy(edge_index)
    num_edges = edge_index.size(1)
    num_droped = int(num_edges*p)
    perm = torch.randperm(num_edges)

    edge_index = edge_index[:, perm[:num_edges-num_droped]]
    
    return edge_index

def feat_drop(x, p=0.2):
    # copy x
    x = copy.deepcopy(x)
    mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x[:, mask] = 0

    return x

