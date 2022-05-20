# 节点与图做对比
# 重写loader

import argparse

import torch
from tqdm import tqdm
import torch.nn.functional as F

from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_scatter import scatter_max, scatter
import torch_geometric as pyg
from copy import deepcopy
from cluster import ClusterData, ClusterLoader

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import math
from torch_geometric.utils import add_remaining_self_loops

import numpy as np
from sklearn.metrics import roc_auc_score,f1_score
from utils import permute_edges, drop_clusters, set_seeds, cluster_graph_aug

parser = argparse.ArgumentParser(description='Yelp (Cluster-GCN)')
parser.add_argument('--seed', type=int, default=777, help='Random seed.')
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=12)

parser.add_argument('--num_partitions', type=int, default=15000)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.0005)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--load_CL', type=int, default=0)
parser.add_argument('--runs', type=int, default=2)

parser.add_argument('--par', type=float, default=0.8, help='对比损失系数')
parser.add_argument('--rate', type=float, default=0.2, help='数据增强扰动概率')
parser.add_argument('--cl', type=str, default='Graphcl',help='Graphcl、GBT、Bgrl')
parser.add_argument('--sample_size', type=float, default=0.2,
                    help='sample size')

args = parser.parse_args()

seed = args.seed
set_seeds(seed)

print(args)
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

# dataset = PygNodePropPredDataset(name='ogbn-products')
dataset = pyg.datasets.Yelp(root="./data/yelp")
data = dataset[0]
data.edge_index,_ = add_remaining_self_loops(data.edge_index)

split_idx = {
    "train": torch.nonzero(data["train_mask"]).squeeze().to(device),
    "valid": torch.nonzero(data["val_mask"]).squeeze().to(device),
    "test": torch.nonzero(data["test_mask"]).squeeze().to(device),
}


# # Convert split indices to boolean masks and add them to `data`.
# for key, idx in split_idx.items():
#     mask = torch.zeros(data.num_nodes, dtype=torch.bool)
#     mask[idx] = True
#     data[f'{key}_mask'] = mask


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
            torch.nn.PReLU(), )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, cluster):
        for conv in self.convs[:-1]:
            out = conv(x, edge_index)
            x = F.relu(out)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = scatter(x, cluster, dim=0, dim_size=cluster.max() + 1, reduce='mean')
        x = self.convs[-1](x, edge_index)
        # torch.log_softmax(x, dim=-1)
        return x, out, g, x

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
            l1 = self.BootstrapLatent(anchor=h1_target, sample=h2_pred)
            l2 = self.BootstrapLatent(anchor=h2_target, sample=h1_pred)

            ret = (l1 + l2) * 0.5

        return ret

def train(model, loader, optimizer, device, epoch, args):
    model.train()
    total_loss = 0
    total_examples = 0
    total_correct = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    i = 0
    if epoch > args.load_CL:
        # print("CL")
        # print("epoch:",epoch)
        for data in loader:
            i = i + 1
            cluster = data.node_cluster
            data_aug = deepcopy(data)
            view1 = cluster_graph_aug(data_aug, args.rate, cluster)
            # view2 = permute_edges(data, args.rate)
            view1 = view1.to(device)
            # view2 = view2.to(device)
            data = data.to(device)
            cluster = cluster.to(device)
            optimizer.zero_grad()

            if args.cl == 'Bgrl':
                _, _, _, x1 = model(view1.x, view1.edge_index, cluster)
                y_pre, _, _, x2 = model(data.x, data.edge_index, cluster)
            else:
                _, x1, g1, _ = model(view1.x, view1.edge_index, cluster)
                y_pre, x2, g2, _ = model(data.x, data.edge_index, cluster)

            # loss_cl = model.graph_node(x1, x2, cluster, node_mask, neg_mask, args.Negsam)
            loss_cl = model.cl_lossaug(x1, x2, cluster, args.cl, args.sample_size)

            out = y_pre[data.train_mask]
            y = data.y[data.train_mask].to(torch.float)

            loss_train = criterion(out, y)

            loss = loss_train + args.par * loss_cl

            loss.backward()
            optimizer.step()

            total_loss += float(loss_train)

            # if i % 100 == 0:
            #     print(f'Batch:{i},loss_train:{loss_train:.6f}, loss_cl:{loss_cl:.6f}, loss:{loss:.6f}')

        loss = total_loss / len(loader)
        print(f'Epoch:{epoch:}, Loss:{loss:.4f}')
        return loss , 0

    else:
        print("original")
        for data in loader:
            i = i + 1

            data = data.to(device)
            cluster = data.node_cluster
            optimizer.zero_grad()
            y_pre, _, _ = model(data.x, data.edge_index, cluster)
            out = y_pre[data.train_mask]
            y = data.y[data.train_mask].to(torch.float)

            loss_train = criterion(out, y)
            loss_train.backward()
            optimizer.step()
            num_examples = data.train_mask.sum().item()
            total_loss += loss_train.item() * num_examples
            total_examples += num_examples

            # if i % 50 == 0:
            #     print(f'Batch:{i},loss_train:{loss_train:.6f}')
        loss = total_loss / len(loader)
        print(f'Epoch:{epoch:}, Loss:{loss:.4f}')
        return loss, 0


def eval_rocauc(y_true, y_pred):
    """
    compute ROC-AUC and AP score averaged across tasks
    """

    if torch is not None and isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if torch is not None and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(
                roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            )

    if len(rocauc_list) == 0:
        raise RuntimeError(
            "No positively labeled data available. Cannot compute ROC-AUC."
        )

    return sum(rocauc_list) / len(rocauc_list)

def eval_f1(y_true, y_pred):
    if torch is not None and isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if torch is not None and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0

    return f1_score(y_true, y_pred, average="micro")

@torch.no_grad()
def test(model, split_idx, x, y, subgraph_loader, device):
    model.eval()

    y_pred = model.inference(x, subgraph_loader, device)

    y_true = y.cpu()

    train_eval = eval_rocauc(y_true[split_idx["train"]], y_pred[split_idx["train"]])
    val_eval = eval_rocauc(y_true[split_idx["valid"]], y_pred[split_idx["valid"]])
    test_eval = eval_rocauc(y_true[split_idx["test"]], y_pred[split_idx["test"]])

    return train_eval, val_eval, test_eval


def main():
    cluster_data = ClusterData(data, num_parts=args.num_partitions,
                               recursive=False, save_dir=dataset.processed_dir)

    loader = ClusterLoader(cluster_data, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=4096, shuffle=False,
                                      num_workers=args.num_workers)

    model = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                 args.num_layers, args.dropout).to(device)

    # evaluator = Evaluator(name='ogbn-products')
    x = data.x.to(device)
    y = data.y.squeeze().to(device)
    vals, tests = [], []
    for run in range(args.runs):
        best_val, final_test = 0, 0

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            loss, acc = train(model, loader, optimizer, device, epoch, args)
            if epoch > 0 and epoch % args.test_freq == 0:

                result = test(model, split_idx, x, y, subgraph_loader, device)
                tra, val, tst = result
                print(f'Epoch:{epoch}, train:{tra:.6f}, val:{val:.6f}, test:{tst:.6f}')
                if val > best_val:
                    best_val = val
                    final_test = tst

        print(f'Run{run} val:{best_val:.6f}, test:{final_test:.6f}')
        vals.append(best_val)
        tests.append(final_test)

    print('')
    print("test:", tests)
    print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals):.6f}")
    print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests):.6f}")
    print(args)


if __name__ == "__main__":
    main()
