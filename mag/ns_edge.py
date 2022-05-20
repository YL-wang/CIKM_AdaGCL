from copy import copy,deepcopy
import argparse
from tqdm import tqdm

import random
import torch
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from utils import permute_edges, drop_clusters, set_seeds, ns_graph_aug
#from logger import Logger

parser = argparse.ArgumentParser(description='OGBN-MAG (SAGE)')
parser.add_argument('--seed', type=int, default=777, help='Random seed.')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=12)

parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--runs', type=int, default=6)


parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=64)

parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--rate', type=float, default=0.2, help='数据增强扰动概率')
parser.add_argument('--par', type=float, default=0.8, help='对比损失系数')

args = parser.parse_args()
seed = args.seed
set_seeds(seed)
print(args)

dataset = PygNodePropPredDataset(name='ogbn-mag')
data = dataset[0]
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-mag')
#logger = Logger(args.runs, args)

# We do not consider those attributes for now.
data.node_year_dict = None
data.edge_reltype_dict = None

#print(data)

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

# Map informations to their canonical type.
x_dict = {}
for key, x in data.x_dict.items():
    x_dict[key2int[key]] = x

num_nodes_dict = {}
for key, N in data.num_nodes_dict.items():
    num_nodes_dict[key2int[key]] = N

# Next, we create a train sampler that only iterates over the respective
# paper training nodes.
paper_idx = local2global['paper']
paper_train_idx = paper_idx[split_idx['train']['paper']]

train_loader = NeighborSampler(edge_index, node_idx=paper_train_idx,
                               sizes=[25, 20], batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)


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

    def forward(self, x, edge_index, edge_type, target_node_type):
        x_src, x_target = x

        out = x_target.new_zeros(x_target.size(0), self.out_channels)

        mask_edge = random.sample(range(0, edge_type.shape[0]), int(edge_type.shape[0] / 5))
        for i in range(self.num_edge_types):
            mask = edge_type == i
            mask[mask_edge] = False
            out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

        for i in range(self.num_node_types):
            mask = target_node_type == i
            out[mask] += self.root_lins[i](x_target[mask])

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

        self.fc1 = torch.nn.Linear(out_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.tau = 0.4

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

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx, n_id=None):
        # Create global node feature matrix.
        if n_id is not None:
            node_type = node_type[n_id]
            local_node_idx = local_node_idx[n_id]

        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]

        return h

    def forward(self, n_id, x_dict, adjs, edge_type, node_type,
                local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx, n_id)
        node_type = node_type[n_id]

        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target node embeddings.
            node_type = node_type[:size[1]]  # Target node types.
            conv = self.convs[i]
            if i == self.num_layers - 1:
                out = x
            x = conv((x, x_target), edge_index, edge_type[e_id], node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1), x, out

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

    def jsd_loss(self, enc1, enc2, pos_mask, neg_mask):
        logits = enc1 @ enc2.t()
        Epos = (np.log(2.) - F.softplus(- logits))
        Eneg = (F.softplus(-logits) + logits - np.log(2.))
        #print("x:",enc1.shape,"g:",enc2.shape,"pos:",pos_mask.shape,"neg:",neg_mask.shape)
        Epos = (Epos * pos_mask).sum() / pos_mask.sum()
        Eneg = (Eneg * neg_mask).sum() / neg_mask.sum()
        return Eneg - Epos

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def cl_lossaug(self, z1, g2, pos_mask, neg_mask):
        h1 = self.projection(z1)
        #h2 = self.projection(z2)
        h1 = F.normalize(h1)
        #h2 = F.normalize(h2)

        ret = model.jsd_loss(h1, g2, pos_mask, neg_mask)

        ret = ret.mean()

        return ret

def graph_em(g, neighbor, cluster):
    neighbor_emb = g[neighbor].to(device)
    g_dim = cluster.max() + 1
    graph_embedding = scatter(neighbor_emb, cluster, dim=0, dim_size=g_dim, reduce='mean')

    return graph_embedding

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

model = RGCN(128, args.hidden_channels, dataset.num_classes, args.num_layers,
             args.dropout, num_nodes_dict, list(x_dict.keys()),
             len(edge_index_dict.keys())).to(device)

# Create global label vector.
y_global = node_type.new_full((node_type.size(0), 1), -1)
y_global[local2global['paper']] = data.y_dict['paper']

# Move everything to the GPU.
x_dict = {k: v.to(device) for k, v in x_dict.items()}
edge_type = edge_type.to(device)
node_type = node_type.to(device)
local_node_idx = local_node_idx.to(device)
y_global = y_global.to(device)


def permute_edges(edge):
    _, edge_num1 = edge[0].edge_index.shape #20
    _, edge_num2 = edge[1].edge_index.shape #25

    permute_num1 = int(edge_num1 * 1 / 5)
    permute_num2 = int(edge_num2 * 1 / 5)

    edge_index1 = edge[0].edge_index.cpu().transpose(0, 1).numpy()
    edge_index2 = edge[1].edge_index.cpu().transpose(0, 1).numpy()


    edge_index1 = edge_index1[np.random.choice(edge_num1, edge_num1 - permute_num1, replace=False)]
    edge_index2 = edge_index2[np.random.choice(edge_num2, edge_num2 - permute_num2, replace=False)]
    edge[0] = edge[0]._replace(edge_index=torch.tensor(edge_index1).transpose_(0, 1)).to(device)
    edge[1] = edge[1]._replace(edge_index=torch.tensor(edge_index2).transpose_(0, 1)).to(device)

    return edge

def train(epoch):
    model.train()

    pbar = tqdm(total=paper_train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        n_id = n_id.to(device)
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()

        out, _, _ = model(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)

        y = y_global[n_id][:batch_size].squeeze()

        loss = F.nll_loss(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / paper_train_idx.size(0)
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
        #logger.add_result(run, result)
        if epoch > 10 and epoch % args.test_freq == 0 or epoch == args.epochs:
            result = test()
            tra, val, tst = result
            print(f'Epoch:{epoch}, train:{tra:.6f}, val:{val:.6f}, test:{tst:.6f}')
            if val > best_val:
                best_val = val
                final_test = tst
    print(f'Run{run} val:{best_val:.6f}, test:{final_test:.6f}')
    vals.append(best_val)
    tests.append(final_test)

print('')
print("test:",tests)
print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals)}:.6f")
print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests):.6f}")
print(args)