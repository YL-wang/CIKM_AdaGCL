#节点与图做对比
# 重写loader

import argparse

import torch
from tqdm import tqdm
import torch.nn.functional as F

from torch_geometric.loader import  NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_scatter import scatter_max, scatter
from torch_geometric.utils import add_remaining_self_loops
from cluster import ClusterData, ClusterLoader

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import math
from copy import deepcopy
import numpy as np
from torch.nn.functional import cosine_similarity

from utils import permute_edges, drop_clusters, set_seeds, cluster_graph_aug

parser = argparse.ArgumentParser(description='OGBN-Products (Cluster-GCN)')
parser.add_argument('--seed', type=int, default=777, help='Random seed.')
parser.add_argument('--device', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=12)

parser.add_argument('--num_partitions', type=int, default=15000)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--test_freq', type=int, default=2)
parser.add_argument('--load_CL', type=int, default=0)
parser.add_argument('--runs', type=int, default=2)

parser.add_argument('--par', type=float, default=0.8, help='对比损失系数')
parser.add_argument('--rate', type=float, default=0.2, help='数据增强扰动概率')

parser.add_argument('--cl', type=str, default='Graphcl',help='Graphcl、GBT、Bgrl')
parser.add_argument('--sample_size', type=float, default=0.1,
                    help='sample size')
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
        self.dropout = dropout
        self._predictor = torch.nn.Sequential(
            torch.nn.Linear(out_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels, momentum=0.01),
            torch.nn.PReLU(),
            torch.nn.Linear(hidden_channels, out_channels),
            torch.nn.BatchNorm1d(out_channels, momentum=0.01),
            torch.nn.PReLU(),)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, cluster):
        for conv in self.convs[:-1]:
            out = conv(x, edge_index)
            x = F.relu(out)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = scatter(x, cluster, dim=0, dim_size=cluster.max()+1, reduce='mean')
        x = self.convs[-1](x, edge_index)

        return torch.log_softmax(x, dim=-1), out, g ,x

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

    def bt_loss(self, h1, h2, lambda_ = None, batch_norm=True, eps=1e-15):
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

    def cl_lossaug(self, z1: torch.Tensor, z2: torch.Tensor, cluster, cl, sample_size):
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
            # anchor = F.normalize(anchor, dim=-1, p=2)
            # sample = F.normalize(sample, dim=-1, p=2)
            # l1 = self.BootstrapLatent(anchor=h1_target, sample=h2_pred)
            # l2 = self.BootstrapLatent(anchor=h2_target, sample=h1_pred)
            ret = 2 - cosine_similarity(h1_pred, h2_target.detach(), dim=-1).mean() - cosine_similarity(h2_pred, h1_target.detach(),
                                                                                             dim=-1).mean()
        return ret


def train(model, loader, optimizer, device, epoch, args):
    model.train()
    total_loss  = 0
    total_examples = 0
    total_correct = 0
    i=0
    if epoch > args.load_CL:
        print("CL")
        print("epoch:",epoch)
        for data in loader:
            i=i+1
            data_aug = deepcopy(data)
            cluster = data.node_cluster
            view1 = cluster_graph_aug(data_aug, args.rate, cluster)
            view1 = view1.to(device)
            data = data.to(device)
            optimizer.zero_grad()
            cluster = cluster.to(device)

            if args.cl == 'Bgrl':
                _, _, _, x1 = model(view1.x, view1.edge_index, cluster)
                y_pre, _, _, x2 = model(data.x, data.edge_index, cluster)
            else:
                _, x1, g1, _ = model(view1.x, view1.edge_index, cluster)
                y_pre, x2, g2, _= model(data.x, data.edge_index, cluster)

            # loss1 = model.jsd_loss(x1, g2, cluster)
            # loss2 = model.jsd_loss(x2, g1, cluster)
            # loss_cl = (loss1+loss2)/10

            loss_cl = model.cl_lossaug(x1, x2, cluster, args.cl, args.sample_size)
            out = y_pre[data.train_mask]
            y = data.y.squeeze(1)[data.train_mask]

            loss_train = F.nll_loss(out, y)

            loss = loss_train + args.par * loss_cl

            loss.backward()
            optimizer.step()

            total_loss += float(loss_train)
        loss = total_loss / len(loader)
        print(f'Epoch:{epoch:}, Loss:{loss:.4f}')
        #print(f'Epoch:{epoch:}, Loss:{total_loss / total_examples:.6f}')
        return 0, 0

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

            # if i % 50 == 0:
            #     print(f'Batch:{i},loss_train:{loss_train:.6f}')
            total_loss += float(loss_train)
        loss = total_loss / len(loader)
        print(f'Epoch:{epoch:}, Loss:{loss:.4f}')
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


def main():
    cluster_data = ClusterData(data, num_parts=args.num_partitions,
                               recursive=False, save_dir=dataset.processed_dir)

    loader = ClusterLoader(cluster_data, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=1024, shuffle=False,
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
            loss, acc = train(model, loader, optimizer, device, epoch, args)
            if epoch > 50 and epoch % args.test_freq == 0 or epoch == args.epochs :

                result = test(model, data, evaluator, subgraph_loader, device)
                tra, val, tst = result
                print(f'Epoch:{epoch}, train:{tra:.6f}, val:{val:.6f}, test:{tst:.6f}')
                if val > best_val:
                    best_val = val
                    final_test = tst

            # elif epoch > 9 and epoch % 10 == 0 or epoch == args.epochs:
            #
            #     result = test(model, data, evaluator, subgraph_loader, device)
            #     tra, val, tst = result
            #     print(f'Epoch:{epoch}, train:{tra}, val:{val}, test:{tst}')
            #     if val > best_val:
            #         best_val = val
            #         final_test = tst

        print(f'Run{run} val:{best_val:.6f}, test:{final_test:.6f}')
        vals.append(best_val)
        tests.append(final_test)

    print('')
    print("test:",tests)
    print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals)}:.6f")
    print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests):.6f}")
    print(args)

if __name__ == "__main__":
    main()
