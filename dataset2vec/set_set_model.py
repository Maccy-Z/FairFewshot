import torch
import torch.nn as nn
from toy_dataset import ToyDataloader
from dataset import Dataloader
import time

torch.manual_seed(0)


class SetSetModel(nn.Module):
    def __init__(self, h_size, out_size):
        super().__init__()

        # f network
        self.f_1 = nn.Linear(2, h_size)
        self.f_2_r = nn.Linear(h_size, h_size)
        self.f_3_r = nn.Linear(h_size, h_size)
        self.f_4_r = nn.Linear(h_size, h_size)
        self.f_5 = nn.Linear(h_size, h_size)

        # g network
        self.g_1 = nn.Linear(h_size, h_size)
        self.g_2 = nn.Linear(h_size, h_size)

        # h network
        self.h_1 = nn.Linear(h_size, h_size)
        self.h_2_r = nn.Linear(h_size, h_size)
        self.h_3_r = nn.Linear(h_size, h_size)
        self.h_4_r = nn.Linear(h_size, h_size)
        self.h_5 = nn.Linear(h_size, out_size)

        self.relu = nn.ReLU()

    def forward(self, xs):
        # x.shape = [BS][x_dim, y_dim, n_samples, 2]
        # Final dim of x is pair (x_i, y_i)
        ys = []
        for x in xs:
            ys.append(self.forward_layers(x))

        ys = torch.stack(ys)

        return ys

    def forward_layers(self, x):
        # f network
        x = self.relu(self.f_1(x))  # [x_dim, y_dim, n_samples, h_size]
        x_r = self.relu(self.f_2_r(x))
        x = x + x_r
        x_r = self.relu(self.f_3_r(x))
        x = x + x_r
        x_r = self.relu(self.f_4_r(x))
        x = x + x_r
        x = self.relu(self.f_5(x))

        x = torch.mean(x, dim=-2)  # [x_dim, y_dim, h_size]

        # g network
        x = self.relu(self.g_1(x))
        x = self.relu(self.g_2(x))
        #
        x = torch.mean(x, dim=(-2, -3))  # [h_size]

        # h network
        x = self.relu(self.h_1(x))
        x_r = self.relu(self.h_2_r(x))
        x = x + x_r
        x_r = self.relu(self.h_3_r(x))
        x = x + x_r
        x_r = self.relu(self.h_4_r(x))
        x = x + x_r
        x = self.relu(self.h_5(x))

        return x


class ModelTrainer:
    def __init__(self, device="cpu"):
        self.gamma = 1

        self.model = SetSetModel(h_size=64, out_size=64).to(device)
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr=5e-4, momentum=0.9)
        # self.dl = ToyDataloader(bs=6, repeat_frac=1 / 2)
        self.dl = Dataloader(bs=8, bs_num_ds=4, device=device)

    def train_loop(self):
        self.dl.steps = 3000
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
        self.dl.steps = 250
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

    def process_batch(self, data_batches: torch.Tensor, ds_labels: torch.Tensor, ):
        # data_batches.shape    = [BS, x_dim, y_dim, num_samples]
        # labels.shape          = [BS]

        bs = len(data_batches)
        assert len(data_batches) == ds_labels.shape[0]
        features = self.model(data_batches)  # [BS, out_dim]

        # Get every distinct combination of features, ignoring diagonals. Split into same ds and different ds.
        # Assume number of same and diff are both nonzero.

        same_diff, diff_diff = [], []
        for i in range(bs):
            for j in range(0, i, 1):
                # If features came from same class
                same = ds_labels[i] == ds_labels[j]
                if same:
                    same_diff.append(features[i] - features[j])
                else:
                    diff_diff.append(features[i] - features[j])

        same_diff = torch.stack(same_diff)
        diff_diff = torch.stack(diff_diff)

        # Similarity, measured between 0 and 1
        same_diff = torch.norm(same_diff, dim=1)
        diff_diff = torch.norm(diff_diff, dim=1)

        # Calculate loss
        # Original paper uses log of exp. Skip exp-log here and get loss directly.
        exp_same_diff = - self.gamma * same_diff
        loss_same = torch.mean(exp_same_diff)
        exp_diff_diff = torch.exp(- self.gamma * diff_diff)
        loss_diff = torch.mean(torch.log((1 - exp_diff_diff)))

        loss = -(loss_same + loss_diff)

        return loss, (same_diff, diff_diff)


if __name__ == "__main__":
    d = torch.device("cpu")

    trainer = ModelTrainer(device=d)

    trainer.train_loop()
    print()
    print()
    trainer.test_loop()
