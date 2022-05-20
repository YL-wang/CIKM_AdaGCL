# 节点与图做对比
# 重写loader

import argparse

import torch
from tqdm import tqdm
import torch.nn.functional as F

from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_scatter import scatter_max, scatter
from torch_geometric.utils import add_remaining_self_loops
from cluster import ClusterData, ClusterLoader

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import math
from copy import deepcopy
import numpy as np

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
parser.add_argument('--rate', type=float, default=0.1, help='数据增强扰动概率')

args = parser.parse_args()

seed = args.seed
set_seeds(seed)

print(args)
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

dataset = PygNodePropPredDataset(name='ogbn-products')
split_idx = dataset.get_idx_split()
data = dataset[0]
data.edge_index, _ = add_remaining_self_loops(data.edge_index)

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
    total_loss, total_test = 0, 0
    total_examples = 0
    total_correct = 0
    i = 0
    list10 = []
    test10 = []

    if epoch > args.load_CL:
        print("CL")
        print("epoch:", epoch)
        for data in loader:
            i = i + 1
            data_aug = deepcopy(data)
            cluster = data.node_cluster
            view1 = cluster_graph_aug(data_aug, args.rate, cluster)
            view1 = view1.to(device)
            data = data.to(device)
            optimizer.zero_grad()
            cluster = cluster.to(device)

            _, x1, g1 = model(view1.x, view1.edge_index, cluster)
            y_pre, x2, g2 = model(data.x, data.edge_index, cluster)

            loss1 = model.jsd_loss(x1, g2, cluster)
            loss2 = model.jsd_loss(x2, g1, cluster)
            loss_cl = (loss1 + loss2) / 10

            out = y_pre[data.train_mask]
            y = data.y.squeeze(1)[data.train_mask]

            loss_train = F.nll_loss(out, y)

            out1 = y_pre[data.test_mask]
            y1 = data.y.squeeze(1)[data.test_mask]
            loss_test = F.nll_loss(out1, y1)

            if epoch % 10 == 0:
                lossa = loss_train.cpu().detach().numpy()
                lossb = loss_test.cpu().detach().numpy()
                list10 = np.append(list10, lossa)
                test10 = np.append(test10, lossb)

            loss = loss_train + args.par * loss_cl

            loss.backward()
            optimizer.step()

            # num_examples = data.train_mask.sum().item()
            # total_loss += loss.item() * num_examples
            # total_examples += num_examples

            # if i % 100 == 0:
            #     print(f'Batch:{i},loss_train:{loss_train:.6f}, loss_cl:{loss_cl:.6f}, loss:{loss:.6f}')
            total_loss += float(loss_train)
            total_test += float(loss_test)
        if epoch % 10 == 0:
            with open("cluster_aug.txt",'a') as f:
                f.write("epoch:")
                f.write(float(epoch))
                f.write('\r\n')
                f.write("train:")
                f.write(list10)
                f.write('\r\n')
                f.write("test:")
                f.write(test10)
                f.write('\n------------------------------------------------')
        loss = total_loss / len(loader)
        losst = total_test / len(loader)

        print(f'Epoch:{epoch:}, Loss:{loss:.4f}')
        return loss, losst

    else:
        print("original")
        for data in loader:
            i = i + 1
            data = data.to(device)
            optimizer.zero_grad()
            cluster = data.node_cluster
            y_pre, _, _ = model(data.x, data.edge_index, cluster)
            out = y_pre[data.train_mask]
            y = data.y.squeeze(1)[data.train_mask]

            loss_train = F.nll_loss(out, y)

            out1 = y_pre[data.test_mask]
            y1 = data.y.squeeze(1)[data.test_mask]
            loss_test = F.nll_loss(out1, y1)
            if epoch % 10 == 0:
                lossa = loss_train.cpu().detach().numpy()
                lossb = loss_test.cpu().detach().numpy()
                list10 = np.append(list10, lossa)
                test10 = np.append(test10, lossb)

            loss_train.backward()
            optimizer.step()

            # if i % 50 == 0:
            #     print(f'Batch:{i},loss_train:{loss_train:.6f}')
            total_loss += float(loss_train)
            total_test += float(loss_test)
        if epoch % 10 == 0:
            with open("cluster.txt",'a') as f:
                f.write("epoch:")
                f.write(float(epoch))
                f.write('\r\n')
                f.write("train:")
                f.write(list10)
                f.write('\r\n')
                f.write("test:")
                f.write(test10)
                f.write('\n------------------------------------------------')
        loss = total_loss / len(loader)
        losst = total_test / len(loader)
        print(f'Epoch:{epoch:}, Loss:{loss:.4f}')
        return loss, losst


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
    losslist, losstest = [], []
    for run in range(args.runs):
        best_val, final_test = 0, 0

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            loss, losst = train(model, loader, optimizer, device, epoch, args)
            losslist = np.append(losslist, loss)
            losstest = np.append(losstest, losst)
            if epoch == args.epochs:
                print('train_loss:', losslist)
                print('test_loss:', losstest)
            if epoch > 40 and epoch % args.test_freq == 0 or epoch == args.epochs:

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
    print("test:", tests)
    print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals)}:.6f")
    print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests):.6f}")
    print(args)


if __name__ == "__main__":
    main()
