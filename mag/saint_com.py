from copy import copy, deepcopy
import argparse
from tqdm import tqdm
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected ,degree, add_remaining_self_loops
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from utils import saint_graph_aug, set_seeds

parser = argparse.ArgumentParser(description='OGBN-MAG (GraphSAINT)')
parser.add_argument('--seed', type=int, default=777, help='Random seed.')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=12)

parser.add_argument('--batch_size', type=int, default=20000)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--walk_length', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=30)

parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.005)

parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--runs', type=int, default=10)

parser.add_argument('--par', type=float, default=0.8, help='对比损失系数')
parser.add_argument('--rate', type=float, default=0.2, help='数据增强扰动概率')
parser.add_argument('--cl', type=str, default='Graphcl',help='Graphcl、GBT、Bgrl')
parser.add_argument('--sample_size', type=float, default=0.1,
                    help='sample size')

args = parser.parse_args()

seed = args.seed
set_seeds(seed)
print(args)

dataset = PygNodePropPredDataset(name='ogbn-mag')
data = dataset[0]
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-mag')

# We do not consider those attributes for now.
data.node_year_dict = None
data.edge_reltype_dict = None

edge_index_dict = data.edge_index_dict

# We need to add reverse edges to the heterogeneous graph.
r, c = edge_index_dict[('author', 'affiliated_with', 'institution')]
edge_index_dict[('institution', 'to', 'author')] = torch.stack([c, r])

r, c = edge_index_dict[('author', 'writes', 'paper')]
edge_index_dict[('paper', 'to', 'author')] = torch.stack([c, r])

r, c = edge_index_dict[('paper', 'has_topic', 'field_of_study')]
edge_index_dict[('field_of_study', 'to', 'paper')] = torch.stack([c, r])

# Convert to undirected paper <-> paper relation.
edge_index = to_undirected(edge_index_dict[('paper', 'cites', 'paper')])
edge_index_dict[('paper', 'cites', 'paper')] = edge_index

# We convert the individual graphs into a single big one, so that sampling
# neighbors does not need to care about different edge types.
# This will return the following:
# * `edge_index`: The new global edge connectivity.
# * `edge_type`: The edge type for each edge.
# * `node_type`: The node type for each node.
# * `local_node_idx`: The original index for each node.
# * `local2global`: A dictionary mapping original (local) node indices of
#    type `key` to global ones.
# `key2int`: A dictionary that maps original keys to their new canonical type.
out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
                 node_type=node_type, local_node_idx=local_node_idx,
                 num_nodes=node_type.size(0))

homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
homo_data.y[local2global['paper']] = data.y_dict['paper']

homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
homo_data.train_mask[local2global['paper'][split_idx['train']['paper']]] = True

train_loader = GraphSAINTRandomWalkSampler(homo_data,
                                           batch_size=args.batch_size,
                                           walk_length=args.num_layers,
                                           num_steps=args.num_steps,
                                           sample_coverage=0,
                                           save_dir=dataset.processed_dir)

# Map informations to their canonical type.
x_dict = {}
for key, x in data.x_dict.items():
    x_dict[key2int[key]] = x

num_nodes_dict = {}
for key, N in data.num_nodes_dict.items():
    num_nodes_dict[key2int[key]] = N


class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types):
        super(RGCNConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.rel_lins = ModuleList([
            Linear(in_channels, out_channels, bias=False)
            for _ in range(num_edge_types)
        ])

        self.root_lins = ModuleList([
            Linear(in_channels, out_channels, bias=True)
            for _ in range(num_node_types)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()

    def forward(self, x, edge_index, edge_type, node_type):
        out = x.new_zeros(x.size(0), self.out_channels)

        for i in range(self.num_edge_types):
            mask = edge_type == i
            out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

        for i in range(self.num_node_types):
            mask = node_type == i
            out[mask] += self.root_lins[i](x[mask])

        return out

    def message(self, x_j, edge_type: int):
        return self.rel_lins[edge_type](x_j)


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types):
        super(RGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Create embeddings for all node types that do not come with features.
        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types))
        self.convs.append(RGCNConv(H, O, self.num_node_types, num_edge_types))
        self._predictor = torch.nn.Sequential(
            torch.nn.Linear(out_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels, momentum=0.01),
            torch.nn.PReLU(),
            torch.nn.Linear(hidden_channels, out_channels),
            torch.nn.BatchNorm1d(out_channels, momentum=0.01),
            torch.nn.PReLU(), )
        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx):
        # Create global node feature matrix.
        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]

        return h

    def forward(self, x_dict, edge_index, edge_type, node_type,
                local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx)

        for i, conv in enumerate(self.convs):
            g = x
            x = conv(x, edge_index, edge_type, node_type)
            if i != self.num_layers - 1:
                out = x
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1), out, g, x

    def inference(self, x_dict, edge_index_dict, key2int):
        # We can perform full-batch inference on GPU.

        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)
        for key, emb in self.emb_dict.items():
            x_dict[int(key)] = emb

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)

        for i, conv in enumerate(self.convs):
            out_dict = {}

            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)

            for keys, adj_t in adj_t_dict.items():
                src_key, target_key = keys[0], keys[-1]
                out = out_dict[key2int[target_key]]
                tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                out.add_(conv.rel_lins[key2int[keys]](tmp))

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])

            x_dict = out_dict

        return x_dict

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def Graphcl(self, x1, x2):

        T = 0.5
        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def bt_loss(self, h1, h2, lambda_=None, batch_norm=True, eps=1e-15):
        batch_size = h1.size(0)
        feature_dim = h1.size(1)

        if lambda_ is None:
            lambda_ = 1. / feature_dim

        if batch_norm:
            z1_norm = (h1 - h1.mean(dim=0)) / (h1.std(dim=0) + eps)
            z2_norm = (h2 - h2.mean(dim=0)) / (h2.std(dim=0) + eps)
            c = (z1_norm.T @ z2_norm) / batch_size
        else:
            c = h1.T @ h2 / batch_size

        off_diagonal_mask = ~torch.eye(feature_dim).bool()
        loss = (1 - c.diagonal()).pow(2).sum()
        loss += lambda_ * c[off_diagonal_mask].pow(2).sum()

        return loss

    def BootstrapLatent(self, anchor, sample):
        batch_size = anchor.size(0)
        pos_mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        neg_mask = 1. - pos_mask
        anchor = F.normalize(anchor, dim=-1, p=2)
        sample = F.normalize(sample, dim=-1, p=2)

        similarity = anchor @ sample.t()
        loss = (similarity * pos_mask).sum(dim=-1)
        return loss.mean()

    def cl_lossaug(self, z1: torch.Tensor, z2: torch.Tensor, cl, sample_size):
        if cl == 'Graphcl':
            node_mask = torch.empty(z1.shape[0], dtype=torch.float32).uniform_(0, 1).cuda()
            node_mask = node_mask < sample_size
            h1 = self.projection(z1)
            h2 = self.projection(z2)
            h1 = h1[node_mask]
            h2 = h2[node_mask]

            loss1 = self.Graphcl(h1, h2)
            ret = loss1

        if cl == 'GBT':
            h1 = self.projection(z1)
            h2 = self.projection(z2)
            loss1 = self.bt_loss(h1, h2)
            ret = loss1.mean()

        if cl == 'Bgrl':
            h1_target, h2_target = z1, z2
            h1_pred = self._predictor(z1)
            h2_pred = self._predictor(z2)
            l1 = self.BootstrapLatent(anchor=h1_target, sample=h2_pred)
            l2 = self.BootstrapLatent(anchor=h2_target, sample=h1_pred)

            ret = (l1 + l2) * 0.5

        return ret

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

model = RGCN(128, args.hidden_channels, dataset.num_classes, args.num_layers,
             args.dropout, num_nodes_dict, list(x_dict.keys()),
             len(edge_index_dict.keys())).to(device)

x_dict = {k: v.to(device) for k, v in x_dict.items()}

def graph_em(g, neighbor, cluster):
    neighbor_emb = g[neighbor]
    g_dim = cluster.max() + 1
    graph_embedding = scatter(neighbor_emb, cluster, dim=0, dim_size=g_dim, reduce='mean')

    return graph_embedding

def train(epoch):
    model.train()
    i = 0
    pbar = tqdm(total=args.num_steps * args.batch_size)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_examples = 0
    for data in train_loader:
        i = i + 1
        neighbor_edge = data.edge_index[:, :(data.edge_index[0] < data.train_mask.sum()).sum()]
        neighbor = neighbor_edge[1].to(device)
        cluster = neighbor_edge[0].to(device)
        node_degree = degree(cluster, data.train_mask.sum())
        _, index = torch.topk(node_degree, 1024)
        data_aug = deepcopy(data)
        x_aug = deepcopy(x_dict)
        view1, x_aug = saint_graph_aug(x_aug, data_aug, args.rate, index, neighbor, cluster, device)
        view1 = view1.to(device)
        data = data.to(device)
        optimizer.zero_grad()

        if args.cl == 'Bgrl':
            _, _, _, x1 = model(x_aug, view1.edge_index, view1.edge_attr, view1.node_type,
                                 view1.local_node_idx)
            out, _, _, x2= model(x_dict, data.edge_index, data.edge_attr, data.node_type,
                                   data.local_node_idx)
        else:
            _, x1, g1, _ = model(x_aug, view1.edge_index, view1.edge_attr, view1.node_type,
                        view1.local_node_idx)
            out, x2, g2, _ = model(x_dict, data.edge_index, data.edge_attr, data.node_type,
                        data.local_node_idx)

        out = out[data.train_mask]
        y = data.y[data.train_mask].squeeze()
        loss_cl = model.cl_lossaug(x1, x2, args.cl, args.sample_size)


        loss_train = F.nll_loss(out, y)
        loss = loss_train + args.par * loss_cl
        loss.backward()
        optimizer.step()

        num_examples = data.train_mask.sum().item()
        total_loss += loss_train.item() * num_examples
        total_examples += num_examples
        pbar.update(args.batch_size)

    pbar.close()
    loss = total_loss / total_examples

    print(f'Epoch:{epoch:}, Loss:{loss:.4f}')

    return loss


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x_dict, edge_index_dict, key2int)
    out = out[key2int['paper']]

    y_pred = out.argmax(dim=-1, keepdim=True).cpu()
    y_true = data.y_dict['paper']

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc


test()  # Test if inference on GPU succeeds.
vals, tests = [], []

for run in range(args.runs):
    best_val, final_test = 0, 0

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        loss = train(epoch)
        torch.cuda.empty_cache()
        if epoch > 0 and epoch % args.test_freq == 0 or epoch == args.epochs:

            result = test()
            tra, val, tst = result
            print(f'Epoch:{epoch}, train:{tra}, val:{val}, test:{tst}')

            if val > best_val:
                best_val = val
                final_test = tst

    print(f'Run{run} val:{best_val}, test:{final_test}')
    vals.append(best_val)
    tests.append(final_test)

print('')
print("test:", tests)
print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals)}")
print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests)}")
print(args)
