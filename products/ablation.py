import argparse

import torch
from tqdm import tqdm
import torch.nn.functional as F

from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_scatter import scatter_max, scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.utils import degree
import math
import random
import numpy as np
from copy import deepcopy
from torch_geometric.utils import add_remaining_self_loops

from utils import saint_graph_aug, set_seeds, mask_nodes

parser = argparse.ArgumentParser(description='OGBN-Products (GraphSaint)')
parser.add_argument('--seed', type=int, default=777, help='Random seed.')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=12)

parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=20000)
parser.add_argument('--walk_length', type=int, default=3)
parser.add_argument('--num_steps', type=int, default=40)

parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--test_freq', type=int, default=3)
parser.add_argument('--load_CL', type=int, default=0)
parser.add_argument('--runs', type=int, default=2)

parser.add_argument('--par', type=float, default=0.8, help='对比损失系数')
parser.add_argument('--rate', type=float, default=0.2, help='数据增强扰动概率')


args = parser.parse_args()

seed = args.seed
set_seeds(seed)

print(args)
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

dataset = PygNodePropPredDataset(name='ogbn-products')
split_idx = dataset.get_idx_split()
data = dataset[0]
data.edge_index,_ = add_remaining_self_loops(data.edge_index)
sampler_data = data
# Convert split indices to boolean masks and add them to `data`.
for key, idx in split_idx.items():
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[idx] = True
    data[f'{key}_mask'] = mask


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.tau = 0.4
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):

        for conv in self.convs[:-1]:
            out = conv(x, edge_index)
            x = F.relu(out)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = x
        x = self.convs[-1](x, edge_index)

        return torch.log_softmax(x, dim=-1), out, g

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


    def jsd_loss(self, enc1, enc2, pos_mask, neg_mask):
        logits = enc1 @ enc2.t()
        Epos = (np.log(2.) - F.softplus(- logits))
        Eneg = (F.softplus(-logits) + logits - np.log(2.))
        # print("x:",enc1.shape,"g:",enc2.shape,"pos:",pos_mask.shape,"neg:",neg_mask.shape)
        Epos = (Epos * pos_mask).sum() / pos_mask.sum()
        Eneg = (Eneg * neg_mask).sum() / neg_mask.sum()
        return Eneg - Epos

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def cl_lossaug(self, z1, g2, pos_mask, neg_mask):
        #h1 = self.projection(z1)
        # h2 = self.projection(z2)
        #h1 = F.normalize(h1)
        # h2 = F.normalize(h2)

        ret = model.jsd_loss(z1, g2, pos_mask, neg_mask)

        ret = ret.mean()

        return ret

def graph_em(g, neighbor, cluster):
    neighbor_emb = g[neighbor]
    g_dim = cluster.max() + 1
    graph_embedding = scatter(neighbor_emb, cluster, dim=0, dim_size=g_dim, reduce='mean')

    return graph_embedding


def train(model, loader, optimizer, device, epoch, args):
    model.train()
    total_loss = total_correct = 0
    i=0
    if epoch > args.load_CL:
        #print("CL")
        for data in loader:
            i=i+1
            neighbor_edge = data.edge_index[:,:(data.edge_index[0]<data.train_mask.sum()).sum()]
            neighbor = neighbor_edge[1].to(device)
            cluster = neighbor_edge[0].to(device)
            node_degree = degree(cluster,data.train_mask.sum())
            _, index = torch.topk(node_degree, 1024)
            data_aug = deepcopy(data)
            #view1 = saint_graph_aug(data_aug, args.rate, index, neighbor, cluster)
            view1 = mask_nodes(data_aug, args.rate)
            view1 = view1.to(device)
            data = data.to(device)
            optimizer.zero_grad()


            _, x1, g1 = model(view1.x, view1.edge_index)
            y_pre, x2, g2 = model(data.x, data.edge_index)

            g1 = graph_em(g1, neighbor, cluster)
            g2 = graph_em(g2, neighbor, cluster)

            y = data.y.squeeze(1)[data.train_mask]
            label = y[index].contiguous().view(-1, 1)
            x1 = x1[index]
            x2 = x2[index]
            g1 = g1[index]
            g2 = g2[index]

            pos_mask = torch.eq(label, label.T).float().to(device)
            neg_mask = 1 - pos_mask

            loss1 = model.cl_lossaug(x1, g2, pos_mask, neg_mask)
            loss2 = model.cl_lossaug(x2, g1, pos_mask, neg_mask)

            loss_cl = (loss1 + loss2) / 10

            out = y_pre[data.train_mask]
            loss_train = F.nll_loss(out, y)

            loss = loss_train + args.par * loss_cl

            loss.backward()
            optimizer.step()
            total_loss += float(loss)

        loss = total_loss / len(loader)


        print(f'Epoch:{epoch:}, Loss:{loss:.4f}')

        return loss,0

    else:
        print("original")
        for data in loader:
            i = i + 1
            data = data.to(device)
            optimizer.zero_grad()
            y_pre, _ = model(data.x, data.edge_index)
            out = y_pre[data.train_mask]
            y = data.y.squeeze(1)[data.train_mask]

            loss_train = F.nll_loss(out, y)
            loss_train.backward()
            optimizer.step()

            if i % 1 == 0:
                print(f'Batch:{i},loss_train:{loss_train}')

        return 0, 0


@torch.no_grad()
def test(model, data, evaluator, subgraph_loader, device):
    model.eval()

    out = model.inference(data.x, subgraph_loader, device)

    y_true = data.y
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[data.train_mask],
        'y_pred': y_pred[data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[data.valid_mask],
        'y_pred': y_pred[data.valid_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_mask],
        'y_pred': y_pred[data.test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc

loader = GraphSAINTRandomWalkSampler(sampler_data,
                                     batch_size=args.batch_size,
                                     walk_length=args.walk_length,
                                     num_steps=args.num_steps,
                                     sample_coverage=0,
                                     save_dir=dataset.processed_dir)

subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                  batch_size=4096, shuffle=False,
                                  num_workers=args.num_workers)

model = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
             args.num_layers, args.dropout).to(device)

evaluator = Evaluator(name='ogbn-products')
vals, tests = [], []
for run in range(args.runs):
    best_val, final_test = 0, 0

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print('epoch:', epoch)
        loss, acc = train(model, loader, optimizer, device, epoch, args)
        if epoch > 100 and epoch % args.test_freq == 0 or epoch == args.epochs:

            result = test(model, data, evaluator, subgraph_loader, device)
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




