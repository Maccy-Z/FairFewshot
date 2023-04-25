import torch
from torch import nn
import random

random.seed(0)
torch.manual_seed(0)


class TextSumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.embedding = nn.Embedding(10, 128)
        self.fc1 = nn.Linear(128, 128)

        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        # x.shape = [BS, set_size]

        x = self.embedding(x)   # = [BS, set_size, 128]
        # Per unit
        x = self.fc1(x)
        x = self.relu(x)

        # Sum set
        x = torch.sum(x, dim=1)  # = [BS, 128]
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x.squeeze()


class NumDataLoader:
    def __init__(self, bs, set_size, epoch_len, device="cpu"):
        self.device = device
        self.bs = bs
        self.set_size = set_size
        self.epoch_len = epoch_len

    def __iter__(self):
        for _ in range(self.epoch_len):
            numbers = torch.randint(0, 10, (self.bs, self.set_size), device=self.device)
            sum_num = torch.sum(numbers, dim=-1)
            yield numbers, sum_num.float()


if __name__ == "__main__":
    dl = NumDataLoader(bs=32, set_size=10, epoch_len=10000)

    model = TextSumModel()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print("Train loop: ")
    running_loss = 0
    for i, (data, label) in enumerate(dl):
        optimizer.zero_grad()

        pred_label = model(data)
        loss = loss_fn(pred_label, label)
        loss.backward()
        optimizer.step()

        running_loss += loss
        if i % 100 == 0:
            print(f'Loss = {running_loss.item() / 100 :.4g}')
            running_loss = 0

    print()
    print("Test loop: ")

    for s in range(3, 100):
        dl = NumDataLoader(bs=1000, set_size=s, epoch_len=1)

        for i, (data, label) in enumerate(dl):

            pred_label = model(data)
            pred_label = torch.round(pred_label)

            acc = (pred_label == label).sum()

            print(f'Set size = {s}, acc = {acc.item() / 1000}')
