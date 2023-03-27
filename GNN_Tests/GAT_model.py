from datasets.dataset import DataLoader
# from GAtt_Func import GATConv_func

import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import numpy as np

class GNN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()

        self.l1 = GATConv(in_dim, hid_dim)
        self.l2 = GATConv(hid_dim, hid_dim)
        self.l3 = GATConv(hid_dim, out_dim)

    def forward(self, x, edge_idx):
        x = self.l1(x, edge_idx)
        x = self.l2(x, edge_idx)
        x = self.l3(x, edge_idx)

        # Classification by taking average over graph
        x = torch.mean(x, dim=(-2))

        return x


def main():
    dl = DataLoader("adult", train=True, bs=1)

    # Sample data to get dimensions
    xs, ys = next(iter(dl))

    in_nodes, out_dim = xs.shape[1], ys.shape[1]
    # Adult is categorical dataset
    model = GNN(1, 16, 2)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Make the edge_index for in_nodes

    """pairs = np.transpose([np.repeat(xs_k, len(ys_k)), np.tile(ys_k, len(xs_k))])
    pairs = pairs.reshape(len(xs_k), len(ys_k), 2)"""

    nodes = torch.arange(in_nodes)
    interleave = torch.repeat_interleave(nodes, in_nodes)
    repeat = torch.tile(nodes, (in_nodes,))
    edge_idx = torch.stack([interleave, repeat])

    losses, accs = [], []
    for i, (xs, ys) in enumerate(dl):
        xs = xs.view(-1, 1)
        ys = ys.to(int).view(-1)

        ys_pred = model(xs, edge_idx).view(1, -1)

        loss = loss_fn(ys_pred, ys)
        loss.backward()

#        optim.step()
        optim.zero_grad()

        losses.append(loss.item())

        predicted_labels = torch.argmax(ys_pred, dim=1)
        accuracy = (predicted_labels == ys).sum().item()
        accs.append(accuracy)

        if i % 200 == 0:
            print()
            print(f'{np.mean(losses) :.4g}')
            print(f'{np.mean(accs) :.4g}')
            losses, accs = [], []


    print()


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
