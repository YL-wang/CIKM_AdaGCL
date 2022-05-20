import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
#import pyro
import random
from torch_geometric.utils import degree
import multiprocessing as mp
from scipy import sparse as sp
from sklearn.preprocessing import normalize, StandardScaler
from cytoolz import curry
from torch_geometric.data import Data, Batch

def cluster_graph_aug(x, data, rate, cluster, device):
    a = np.array([])
    x[3] = x[3].detach().cpu().numpy()
    index = 0
    node_degree = degree(data.edge_index[0])
    for i in range(cluster.max()+1):
        local = cluster[cluster==i].size()[0]
        drop_num = int(local * rate)
        _, idx= torch.topk(node_degree[index:index+local], drop_num, largest=False)
        a= np.append(a, idx+index)
        index = index+local
    a = a.astype(int)
    node_num = x[3].shape[0]
    feat_dim = x[3].shape[1]
    x[3][a] = torch.zeros((a.shape[0], feat_dim))
    x[3] = torch.from_numpy(x[3]).to(device)
    return data, x

def cluster_graph_aug1(x, data, rate, cluster):
    a = np.array([])
    b = np.array([])
    index = 0
    node_degree = degree(data.edge_index[0])
    for i in range(cluster.max()+1):
        local = cluster[cluster==i].size()[0]
        drop_num = int(local * rate)
        _, idx= torch.topk(node_degree[index:index+local], drop_num, largest=False)
        a= np.append(a, idx+index)
        index = index+local
    node_num, _ = x[3].size()
    _, edge_num = data.edge_index.size()
    edge_index = data.edge_index.numpy()
    idx_nondrop = [n for n in range(index) if not n in a]
    idx_nondrop = np.array(idx_nondrop)
    edge_attr = data.edge_attr

    for j in idx_nondrop:
        id1 = [i for i, x in enumerate(edge_index[0]) if x == j]
        b = np.append(b, id1)
        id2 = [i for i, x in enumerate(edge_index[1]) if x == j]
        b = np.append(b, id2)
    b = b.set()
    data.edge_attr = edge_attr
    data.edge_index = edge_index
    return data, x

def saint_graph_aug(x, data, rate, index, neighbor, cluster, device):
    a = np.array([])
    index = index.detach().cpu().numpy()
    neighbor = neighbor.detach().cpu().numpy()
    cluster = cluster.detach().cpu().numpy()
    x[3] = x[3].detach().cpu().numpy()
    node_num = x[3].shape[0]
    feat_dim = x[3].shape[1]
    mask_num = int(index.shape[0]*rate)

    idx_add = np.random.choice(index, mask_num)
    for i in range(mask_num):
        idx = np.argwhere(cluster == idx_add[i])
        a = np.append(a, idx)
    a = neighbor[a.astype(int)]
    a = np.unique(a)
    x[3][a] = torch.zeros((a.shape[0], feat_dim))
    x[3] = torch.from_numpy(x[3]).to(device)
    return data, x

def ns_graph_aug(x, data, rate, neighbor, cluster, device):
    a = np.array([])
    neighbor = neighbor.detach().cpu().numpy()
    cluster = cluster.detach().cpu().numpy()
    x[3] = x[3].detach().cpu().numpy()
    node_num = x[3].shape[0]
    feat_dim = x[3].shape[1]
    mask_num = int(1024*rate)

    idx_add = np.random.choice(1024, mask_num)
    for i in range(mask_num):
        idx = np.argwhere(cluster == idx_add[i])
        a = np.append(a, idx)
    a = neighbor[a.astype(int)]
    a = np.unique(a)
    x[3][a] = torch.zeros((a.shape[0], feat_dim))
    x[3] = torch.from_numpy(x[3]).to(device)
    return data, x

def ns_graph_aug1(edge, device):
    _, edge_num1 = edge[0].edge_index.shape  # 4
    _, edge_num2 = edge[1].edge_index.shape  # 8
    _, edge_num3 = edge[2].edge_index.shape  # 24

    permute_num1 = int(edge_num1 * 1 / 2)
    permute_num2 = int(edge_num2 * 1 / 4)
    permute_num3 = int(edge_num3 * 1 / 6)

    edge_index1 = edge[0].edge_index.cpu().transpose(0, 1).numpy()
    edge_index2 = edge[1].edge_index.cpu().transpose(0, 1).numpy()
    edge_index3 = edge[2].edge_index.cpu().transpose(0, 1).numpy()

    edge_index1 = edge_index1[np.random.choice(edge_num1, edge_num1 - permute_num1, replace=False)]
    edge_index2 = edge_index2[np.random.choice(edge_num2, edge_num2 - permute_num2, replace=False)]
    edge_index3 = edge_index3[np.random.choice(edge_num3, edge_num3 - permute_num3, replace=False)]
    edge[0] = edge[0]._replace(edge_index=torch.tensor(edge_index1).transpose_(0, 1)).to(device)
    edge[1] = edge[1]._replace(edge_index=torch.tensor(edge_index2).transpose_(0, 1)).to(device)
    edge[2] = edge[2]._replace(edge_index=torch.tensor(edge_index3).transpose_(0, 1)).to(device)

    return edge

def drop_nodes(data, rate):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * rate)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    return data


def permute_edges(data, rate):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * rate)
    edge_index = data.edge_index.transpose(0, 1).numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))

    edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    return data


def subgraph(data, rate):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * (1-rate))

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    return data


def mask_nodes(data, rate):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * rate)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    #data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)
    data.x[idx_mask] = torch.zeros((mask_num, feat_dim))
    return data

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    #pyro.set_rng_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def drop_clusters(data):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()

    drop = random.choice([i for i in range(1, data.node_cluster.max())])
    idx_drop = data.node_cluster==drop
    idx_drop = np.where(idx_drop)[0]

    # idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    # idx_dict = {idx_nondrop[n]: n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    return data
