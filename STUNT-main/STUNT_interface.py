import time
import sys

from data.dataset import get_meta_dataset
from utils import Logger, set_random_seed, cycle
from torchmeta.utils.prototype import get_prototypes
from train.metric_based import get_accuracy
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

device = torch.device("cpu")


class MLPProto(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes):
        super(MLPProto, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
#
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_sizes, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_sizes, hidden_sizes, bias=True)
        )

    def forward(self, inputs):
        if self.training:

            embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))
            return embeddings.view(*inputs.shape[:2], -1)
        else:
            embeddings = self.encoder(inputs)
            return embeddings


def protonet_step(step, model, optimizer, batch):
    stime = time.time()

    train_inputs, train_targets = batch['train']
    num_ways = len(set(list(train_targets[0].numpy())))
    test_inputs, test_targets = batch['test']

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    train_embeddings = model(train_inputs)

    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)
    test_embeddings = model(test_inputs)

    prototypes = get_prototypes(train_embeddings, train_targets, num_ways)

    squared_distances = torch.sum((prototypes.unsqueeze(2)
                                   - test_embeddings.unsqueeze(1)) ** 2, dim=-1)
    loss = F.cross_entropy(-squared_distances, test_targets)

    """ outer gradient step """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = get_accuracy(prototypes, test_embeddings, test_targets).item()

    if step % 50 == 0:
        print()
        print(time.time() - stime)
        print((loss.item()))
        print(acc)


def main(device):
    lr = 0.001
    model_size = (105, 1024, 1024)
    steps = 100
    num_shots = 1
    seed = 0
    """ fixing randomness """
    set_random_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    """ define dataset and dataloader """
    train_set, val_set = get_meta_dataset(num_shots=num_shots, seed=seed, dataset="income")

    train_loader = train_set

    """ Initialize model, optimizer, loss_scalar (for amp) and scheduler """
    model = MLPProto(*model_size).to(device)
    model.train()

    params = model.parameters()
    optimizer = optim.Adam(params, lr=lr)
    print("Starting training ")
    for step in range(1, steps + 1):
        train_batch = next(train_loader)

        protonet_step(step, model, optimizer, train_batch)

        torch.save(model, "/home/maccyz/Documents/STUNT-main/logs/model.pt")

def eval():
    data_name = "income"
    shot_num = 1
    load_path = ''
    seed = 0
    input_size = 105
    output_size = 2
    hidden_dim = 1024


    model = torch.load("/home/maccyz/Documents/STUNT-main/logs/model.pt").cpu().eval()
    # model = MLPProto(input_size, hidden_dim, hidden_dim)
    # model.load_state_dict(load_model.state_dict())

    train_x = np.load('./data/income/xtrain.npy')
    train_y = np.load('./data/income/ytrain.npy')
    test_x = np.load('./data/income/xtest.npy')
    test_y = np.load('./data/income/ytest.npy')
    train_idx = np.load('./data/income/index{}/train_idx_{}.npy'.format(shot_num, seed))

    few_train = model(torch.tensor(train_x[train_idx]).float())
    support_x = few_train.detach().numpy()
    support_y = train_y[train_idx]

    few_test = model(torch.tensor(test_x).float())
    query_x = few_test.detach().numpy()
    query_y = test_y

    def get_accuracy(prototypes, embeddings, targets):
        sq_distances = torch.sum((prototypes.unsqueeze(1)
                                  - embeddings.unsqueeze(2)) ** 2, dim=-1)
        _, predictions = torch.min(sq_distances, dim=-1)
        return torch.mean(predictions.eq(targets).float()) * 100.

    train_x = torch.tensor(support_x.astype(np.float32)).unsqueeze(0)
    train_y = torch.tensor(support_y.astype(np.int64)).unsqueeze(0).type(torch.LongTensor)
    val_x = torch.tensor(query_x.astype(np.float32)).unsqueeze(0)
    val_y = torch.tensor(query_y.astype(np.int64)).unsqueeze(0).type(torch.LongTensor)
    prototypes = get_prototypes(train_x, train_y, output_size)
    acc = get_accuracy(prototypes, val_x, val_y).item()

    print(seed, acc)


if __name__ == "__main__":
    """ argument define """
    # eval()
    # exit(2)
    main("cpu")
