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

parser = argparse.ArgumentParser(description='Amazon (Cluster-GCN)')
parser.add_argument('--seed', type=int, default=777, help='Random seed.')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=12)

parser.add_argument('--num_partitions', type=int, default=15000)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--test_freq', type=int, default=2)
parser.add_argument('--load_CL', type=int, default=1000)
parser.add_argument('--runs', type=int, default=6)

parser.add_argument('--par', type=float, default=0.8, help='对比损失系数')
parser.add_argument('--rate', type=float, default=0.2, help='数据增强扰动概率')

args = parser.parse_args()

seed = args.seed
set_seeds(seed)

print(args)
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

#dataset = PygNodePropPredDataset(name='ogbn-products')
#dataset = pyg.datasets.Yelp(root="./data/yelp")
# https://drive.google.com/drive/folders/1uc76iCxBnd0ntNliosYDHHUc_ouXv9Iv  amazon下载
#dataset = pyg.datasets.AmazonProducts(root="./data/amazon_products")
dataset = pyg.datasets.AmazonProducts(root="/root/autodl-tmp/data/amazon")
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
        return x, out, g

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

    def jsd_loss(self, enc1, enc2, indices):
        pos_mask = torch.eye(enc1.shape[0], enc2.shape[0], device=enc1.device)
        if enc1.shape[0] != enc2.shape[0]:
            pos_mask = pos_mask[indices]
        neg_mask = 1. - pos_mask
        logits = enc1 @ enc2.t()
        Epos = (np.log(2.) - F.softplus(- logits))
        Eneg = (F.softplus(-logits) + logits - np.log(2.))
        Epos = (Epos * pos_mask).sum() / pos_mask.sum()
        Eneg = (Eneg * neg_mask).sum() / neg_mask.sum()
        return Eneg - Epos


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

            _, x1, g1 = model(view1.x, view1.edge_index, cluster)
            # _, x2 = model(view2.x, view2.edge_index)
            y_pre, x2, g2 = model(data.x, data.edge_index, cluster)

            # loss_cl = model.graph_node(x1, x2, cluster, node_mask, neg_mask, args.Negsam)
            loss1 = model.jsd_loss(x1, g2, cluster)
            loss2 = model.jsd_loss(x2, g1, cluster)
            loss_cl = (loss1 + loss2) / 10

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
        return 0, 0

    else:
        print("original")
        for data in loader:
            i = i + 1
            data = permute_edges(data, args.rate)
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
            total_loss += loss_train.item()


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
            if epoch > 45 and epoch % args.test_freq == 0:

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
