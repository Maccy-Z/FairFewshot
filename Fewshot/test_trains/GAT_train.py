from datasets.dataset import DataLoader
# from GAtt_Func import GATConv_func

import torch
from torch_geometric.nn import GATConv, GCNConv
import numpy as np
import torch.nn as nn


class GNN(torch.nn.Module):
    def __init__(self, hid_dim, out_dim, in_nodes):
        super().__init__()

        self.x_pos = torch.nn.Parameter(torch.zeros(in_nodes).view(-1, 1).to(torch.float32))

        heads = 2
        hid_out = int(hid_dim / heads)

        self.l1 = GATConv(2, hid_out, heads=heads)
        self.l2 = GATConv(hid_dim, hid_out, heads=heads)
        # self.l3 = GATConv(hid_dim, hid_out, heads=heads)
        self.out = GATConv(hid_dim, out_dim, heads=1)

        self.lin1 = nn.Linear(2, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)


    def forward(self, x, edge_idx):
        x = torch.cat([x, self.x_pos], dim=-1)

        #x = self.lin1(x)
        # x = F.relu(x)
        # x = self.lin2(x)

        x = self.l1(x, edge_idx)

        x = self.l2(x, edge_idx)
        # x = self.l3(x, edge_idx)
        x = self.out(x, edge_idx)

        # Classification by taking average over graph
        x = torch.mean(x, dim=(-2))

        return x


# Generate additional fixed embeddings / graph
def graph_and_embeddings(in_nodes):
    # Densely connected graph
    nodes = torch.arange(in_nodes)
    interleave = torch.repeat_interleave(nodes, in_nodes)
    repeat = torch.tile(nodes, (in_nodes,))
    edge_idx = torch.stack([interleave, repeat])

    # Positional embeddings
    x_pos = torch.arange(in_nodes).to(torch.float32).view(-1, 1)
    x_pos = x_pos - in_nodes / 2

    return edge_idx, x_pos


def test(model, device="cpu"):
    model.eval()

    dl = DataLoader("adult", train=False, bs=1, device=device)

    # Sample data to get dimensions
    xs, ys = next(iter(dl))

    in_nodes, out_dim = xs.shape[1], ys.shape[1]

    edge_idx, x_pos = graph_and_embeddings(in_nodes)
    edge_idx, x_pos = edge_idx.to(device), x_pos.to(device)

    accs = []

    for i, (xs, ys) in enumerate(dl):
        if ys == 0 and i % 3 != 0:
            continue

        xs = xs.view(-1, 1)
        #

        ys = ys.to(int).view(-1)

        with torch.no_grad():
            ys_pred = model(xs, edge_idx).view(1, -1)

        predicted_labels = torch.argmax(ys_pred, dim=1)
        accuracy = (predicted_labels == ys).sum().item()
        accs.append(accuracy)

    print()
    print("Test Results: ")
    print(f'accuracy: {np.mean(accs) * 100 :.4g}%')


def train(device="cpu"):
    dl = DataLoader("adult", train=True, bs=1, device=device)

    # Sample data to get dimensions
    xs, ys = next(iter(dl))

    in_nodes, out_dim = xs.shape[1], ys.shape[1]
    model = GNN(32, 2, in_nodes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Adult is categorical dataset
    loss_fn = torch.nn.CrossEntropyLoss()

    edge_idx, x_pos = graph_and_embeddings(in_nodes)
    edge_idx, x_pos = edge_idx.to(device), x_pos.to(device)

    accs = []

    for epoch in range(2):
        for i, (xs, ys) in enumerate(dl):
            # Balance the dataset
            if ys == 0 and i % 3 != 0:
                continue

            xs = xs.view(-1, 1)

            ys = ys.to(int).view(-1)

            ys_pred = model(xs, edge_idx).view(1, -1)

            loss = loss_fn(ys_pred, ys)
            loss.backward()

            optim.step()
            optim.zero_grad()

            predicted_labels = torch.argmax(ys_pred, dim=1)
            accuracy = (predicted_labels == ys).sum().item()
            accs.append(accuracy)

            if i % 500 == 0:
                print(f'accuracy: {np.mean(accs) * 100 :.2f}%')

                # print(predicted_labels, ys)
                accs = []

    return model


if __name__ == "__main__":
    torch.manual_seed(5)

    dev = torch.device("cpu")
    m = train(device=dev)
    test(m, device=dev)
