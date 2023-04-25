import torch
import torch.nn as nn
from datasets.dataset import DataLoader
import numpy as np

torch.manual_seed(0)


class TransformerLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.model = nn.TransformerEncoderLayer(d_model=dim, nhead=2, dropout=0, dim_feedforward=128)
        # self.model.norm1 = nn.Identity()
        # self.model.norm2 = nn.Identity()

    def forward(self, x):
        return self.model(x)


class Transformer(torch.nn.Module):
    def __init__(self, in_nodes, hid_dim, out_dim):
        super().__init__()
        self.hid_dim = hid_dim

        self.x_pos = torch.nn.Parameter(torch.zeros(in_nodes, hid_dim-1).to(torch.float32))

        self.l1 = TransformerLayer(hid_dim)
        self.l2 = TransformerLayer(hid_dim)
        self.l3 = TransformerLayer(hid_dim)

        self.out = nn.Linear(hid_dim, out_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.cat([x, self.x_pos], dim=-1)

        x = x.view(1, -1, self.hid_dim)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        x = torch.mean(x, dim=-2)

        x = self.out(x)

        # Classification by taking average over graph
        x = torch.mean(x, dim=(-2))

        return x


def test(model, device="cpu", balance=True):
    dl = DataLoader("adult", train=False, bs=1, device=device)

    # Sample data to get dimensions

    accs = []
    for i, (xs, ys) in enumerate(dl):
        if balance:
            # Balance the dataset
            if ys == 0 and i % 3 != 0:
                continue

        xs = xs.view(-1, 1)

        ys = ys.to(int).view(-1)

        with torch.no_grad():
            ys_pred = model(xs).view(1, -1)

        predicted_labels = torch.argmax(ys_pred, dim=1)
        accuracy = (predicted_labels == ys).sum().item()
        accs.append(accuracy)

    print()
    print("Test Results: ")
    print(f'accuracy: {np.mean(accs) * 100 :.4g}%')


def train(device="cpu", balance=False):
    dl = DataLoader("adult", train=True, bs=1, device=device)

    # Sample data to get dimensions
    xs, ys = next(iter(dl))

    in_nodes, out_dim = xs.shape[1], ys.shape[1]

    model = Transformer(in_nodes, 16, 2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Adult is categorical dataset
    loss_fn = torch.nn.CrossEntropyLoss()

    accs = []
    for epoch in range(1):
        print(f'Epoch {epoch}: ')
        for i, (xs, ys) in enumerate(dl):
            if balance:
                # Balance the dataset
                if ys == 0 and i % 3 != 0:
                    continue
            xs = xs.view(-1, 1)

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
                print(f'Step {i}, accuracy: {np.mean(accs) * 100 :.2f}%')

                # print(predicted_labels, ys)
                accs = []

    return model


if __name__ == "__main__":
    torch.manual_seed(0)

    dev = torch.device("cpu")

    print("Training started")
    m = train(device=dev)
    print("Training Complete")
    test(m, device=dev)
