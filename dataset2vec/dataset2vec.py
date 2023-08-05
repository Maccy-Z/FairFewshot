import torch
import torch.nn as nn
from d2v_dataset import Dataloader
import time
import numpy as np
import random


class ResBlock(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_blocks, out_relu=True):
        super().__init__()
        self.out_relu = out_relu

        self.res_modules = nn.ModuleList([])
        self.lin_in = nn.Linear(in_size, hid_size)

        for _ in range(n_blocks - 2):
            self.res_modules.append(nn.Linear(hid_size, hid_size))

        self.lin_out = nn.Linear(hid_size, out_size)
        self.lin_out.bias.data.fill_(0)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_in(x)
        for layer in self.res_modules:
            x_r = self.act(layer(x))
            x = x + x_r

        if self.out_relu:
            out = self.act(self.lin_out(x))
        else:
            out = self.lin_out(x)
        return out, x


class Dataset2Vec(nn.Module):
    def __init__(self, h_size, out_size, n_blocks):
        super().__init__()

        # f network
        self.fs = ResBlock(2, h_size, h_size, n_blocks=n_blocks[0])

        # g network
        self.gs = ResBlock(h_size, h_size, h_size, n_blocks=n_blocks[1])

        # h network
        self.hs = ResBlock(h_size, h_size, out_size, n_blocks=n_blocks[2], out_relu=False)


    def forward(self, xs):
        # x.shape = [BS][x_dim, y_dim, n_samples, 2]
        # Final dim of x is pair (x_i, y_i)
        ys = []
        for x in xs:
            ys.append(self.forward_layers(x)[0])
        ys = torch.stack(ys)

        return ys

    def forward_layers(self, x):
        # x.shape = # [x_dim, y_dim, n_samples, h_size]
        # f network
        x, _ = self.fs(x)

        x = torch.mean(x, dim=-2)  # [x_dim, y_dim, h_size]
        x_save = x

        # g network
        x, _ = self.gs(x)
        x = torch.mean(x, dim=(-2, -3))  # [h_size]

        # h network
        x, _ = self.hs(x)

        return x, x_save


class ModelTrainer:
    def __init__(self, device="cpu"):
        self.gamma = 1

        h_size, out_size, n_blocks = 64, 64, [4, 2, 4]
        self.params = (h_size, out_size, n_blocks)

        self.model = Dataset2Vec(h_size=h_size, out_size=out_size, n_blocks=n_blocks).to(device)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.dl = Dataloader(bs=12, bs_num_ds=6, device=device, nsamples=16, min_ds_len=250)

    def train_loop(self):
        self.dl.steps = 15000
        self.dl.train(True)

        st = time.time()
        for i, (data, ds_label) in enumerate(self.dl):

            self.optimiser.zero_grad()
            loss, _ = self.process_batch(data, ds_label)
            loss.backward()
            self.optimiser.step()

            if i % 50 == 0:
                print(f'step {i}, time = {time.time() - st :.2g}s, loss = {loss.item() :.4g}')
                st = time.time()

    def test_loop(self):
        self.dl.steps = 2000
        self.dl.train(False)

        with torch.no_grad():
            same_accs, diff_accs = [], []
            for data, ds_label in self.dl:
                _, (same_diff, diff_diff) = self.process_batch(data, ds_label)

                same_diff, diff_diff = torch.exp(- self.gamma * same_diff), torch.exp(- self.gamma * diff_diff)
                # print(same_diff, diff_diff)

                # Define accurate if similarity is more/less than 0.5
                same_acc = same_diff > 0.5
                diff_acc = diff_diff < 0.5

                same_accs.append(same_acc)
                diff_accs.append(diff_acc)

        same_accs = torch.mean(torch.stack(same_accs).float())
        diff_accs = torch.mean(torch.stack(diff_accs).float())

        same_accs, diff_accs = same_accs.item(), diff_accs.item()
        print(f'{same_accs = :.4g}, {diff_accs = :.4g}')

    def process_batch(self, data_batches: list[torch.Tensor], ds_labels: torch.Tensor):
        # data_batches.shape    = [BS, x_dim, y_dim, num_samples]
        # labels.shape          = [BS]

        bs = len(data_batches)
        assert len(data_batches) == ds_labels.shape[0]
        features = self.model(data_batches)  # [BS, out_dim]

        # Calculate distance for pairs of features.

        same_diff, diff_diff = [], []
        for i in range(bs):
            js = torch.arange(i)
            same = ds_labels[i] == ds_labels[js]
            same_idx = torch.nonzero(same, as_tuple=True)[0]
            diff_idx = torch.nonzero(same == 0, as_tuple=True)[0]

            diffs = features[i] - features[js]

            same_diff.append(diffs[same_idx])
            diff_diff.append(diffs[diff_idx])

        same_diff = torch.cat(same_diff)
        diff_diff = torch.cat(diff_diff)

        # Similarity, measured between 0 and 1
        same_diff = torch.norm(same_diff, dim=1)
        diff_diff = torch.norm(diff_diff, dim=1)

        # Calculate loss
        # Original paper uses log of exp. Skip exp-log here and get loss directly.
        exp_same_diff = - self.gamma * same_diff
        loss_same = torch.mean(exp_same_diff)
        exp_diff_diff = torch.exp(- self.gamma * diff_diff)
        loss_diff = torch.mean(torch.log((1 - exp_diff_diff)))

        loss = -(loss_same + 1.25 * loss_diff)

        return loss, (same_diff, diff_diff)

    def save_model(self):
        torch.save({"state_dict": self.model.state_dict(),
                    "params": self.params}, "./dataset2vec/model")

    def load_model(self):
        self.model = torch.load("./dataset2vec/model")

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    
    d = torch.device("cpu")

    trainer = ModelTrainer(device=d)

    trainer.train_loop()
    print()
    print()
    trainer.test_loop()

    trainer.save_model()


"""# 64 32
# same_accs = 0.968, diff_accs = 0.8468
# 64 64
# same_accs = 0.97, diff_accs = 0.863
# 32 32 2xf
# same_accs = 0.968, diff_accs = 0.8528

# no res block
# same_accs = 0.932, diff_accs = 0.8837
# no res block size = 3
# same_accs = 0.9232, diff_accs = 0.8562
# no res block size = 7
# same_accs = 0.9472, diff_accs = 0.839
# 1 res block size = 5
# same_accs = 0.9168, diff_accs = 0.875

# > 500 samples train
# no res block
# same_accs = 0.9296, diff_accs = 0.8464
# 2 f res blocks
# same_accs = 0.9064, diff_accs = 0.8592
# 1 f res block
# same_accs = 0.9176, diff_accs = 0.8421
# 3 g layers
# same_accs = 0.9216, diff_accs = 0.8688
# 3 g layers no res blocks BS=12 out_size=48
# same_accs = 0.9183, diff_accs = 0.8818
# same_accs = 0.9127, diff_accs = 0.8702
# same_accs = 0.932, diff_accs = 0.8602
# same_accs = 0.8872, diff_accs = 0.8643
# 3 g layers no res blocks BS=12 out_size=32
# same_accs = 0.9072, diff_accs = 0.8557
# 3 g layers no res blocks BS=10 out_size=64
# same_accs = 0.9064, diff_accs = 0.873
# 3 g layers no res blocks BS=14 out_size=64
# same_accs = 0.9274, diff_accs = 0.8541
# 3 g layers no res blocks BS=14
# same_accs = 0.9037, diff_accs = 0.8462
# BS=12
# same_accs = 0.9027, diff_accs = 0.8444


# 7 blocks
# same_accs = 0.9302, diff_accs = 0.855
# 11
# same_accs = 0.9018, diff_accs = 0.8501
# 7-5-7
# same_accs = 0.9268, diff_accs = 0.8566
# Zero bias on output
# same_accs = 0.9372, diff_accs = 0.8493
# Above, train 6000 steps
# same_accs = 0.9328, diff_accs = 0.8654
# Train 9k steps
# same_accs = 0.922, diff_accs = 0.9028
# Train 12k steps
# same_accs = 0.938, diff_accs = 0.8697
# Train 15k steps
# same_accs = 0.9368, diff_accs = 0.8871
"""

# Binarised data
# h_size, out_size, n_blocks = 64, 64, [5, 5, 5]
# same_accs = 0.9543, diff_accs = 0.8249
# 64, 64, [7, 7, 7]
# same_accs = 0.924, diff_accs = 0.8443
# 64, 64, [3, 3, 3]
# same_accs = 0.9313, diff_accs = 0.844
# 64, 32, [3, 3, 3]
# same_accs = 0.9307, diff_accs = 0.84
# 64, 64, [5, 5, 5]
# same_accs = 0.9543, diff_accs = 0.8249
# 64, 64, [4, 4, 4]
# same_accs = 0.9518, diff_accs = 0.8377

# Increase loss on diff to 1.5x
# 64, 64, [4, 4, 4]
# same_accs = 0.9032, diff_accs = 0.8889
# 64, 64, [3, 3, 3]
#  same_accs = 0.8852, diff_accs = 0.8968
# 64, 64, [7, 7, 7]
# 0.9063, diff_accs = 0.8945
# 64, 64, [5, 5, 5]
# same_accs = 0.9207, diff_accs = 0.8811

# 9k steps
# 64, 64, [5, 5, 5]
# same_accs = 0.9348, diff_accs = 0.8863
# 64, 64, [3, 3, 3]
# same_accs = 0.9294, diff_accs = 0.8857
# 64, 64, [7, 7, 7]
# same_accs = 0.9368, diff_accs = 0.8914
# 64, 64, [5, 3, 3]
# same_accs = 0.9233, diff_accs = 0.8905
# 64, 64, [5, 3, 5]
# same_accs = 0.9253, diff_accs = 0.8976
# 64, 64, [4, 2, 4]
# same_accs = 0.9393, diff_accs = 0.8864

# Increase loos on diff to 2x
# same_accs = 0.9143, diff_accs = 0.9129
# same_accs = 0.8864, diff_accs = 0.9314
# Train 12 k
# same_accs = 0.9289, diff_accs = 0.9027
# same_accs = 0.9097, diff_accs = 0.9218
# Train 15k
# same_accs = 0.9123, diff_accs = 0.919

# Standardise train procedure
# same_accs = 0.8217, diff_accs = 0.849
# same_accs = 0.8215, diff_accs = 0.8626