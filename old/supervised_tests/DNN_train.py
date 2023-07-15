from datasets.old.dataset import DataLoader
# from GAtt_Func import GATConv_func

import torch
import torch.nn as nn
import numpy as np


class DNN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()

        # x_pos = torch.arange(in_nodes).to(torch.float32).view(-1, 1)
        # self.x_pos = x_pos - in_nodes / 2

        self.l1 = nn.Linear(in_dim, hid_dim)
        self.l2 = nn.Linear(hid_dim, hid_dim)
        self.l3 = nn.Linear(hid_dim, hid_dim)
        self.out = nn.Linear(hid_dim, out_dim)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        # x = self.l3(x, edge_idx)
        x = self.out(x)

        # Classification by taking average over graph
        x = torch.mean(x, dim=(-2))

        return x


def test(model, device="cpu"):
    dl = DataLoader("adult", train=False, bs=1, device=device)

    # Sample data to get dimensions

    accs = []

    for i, (xs, ys) in enumerate(dl):
        if ys == 0 and i % 3 != 0:
            continue

        ys = ys.to(int).view(-1)

        with torch.no_grad():
            ys_pred = model(xs).view(1, -1)

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
    model = DNN(in_nodes, 64, 2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Adult is categorical dataset
    loss_fn = torch.nn.CrossEntropyLoss()

    accs = []

    for epoch in range(2):
        for i, (xs, ys) in enumerate(dl):
            # Balance the dataset
            if ys == 0 and i % 3 != 0:
                continue

            ys = ys.to(int).view(-1)

            ys_pred = model(xs).view(1, -1)

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
    torch.manual_seed(0)

    dev = torch.device("cpu")
    m = train(device=dev)
    test(m, device=dev)
