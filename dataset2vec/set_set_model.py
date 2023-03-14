import torch
import torch.nn as nn
from toy_dataset import ToyDataloader

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

    def forward(self, x):
        # x.shape = [BS, x_dim, y_dim, n_samples, 2]
        # Final dim of x is pair (x_i, y_i)

        # f network
        x = self.relu(self.f_1(x))  # [BS, x_dim, y_dim, n_samples, h_size]
        x_r = self.relu(self.f_2_r(x))
        x = x + x_r
        x_r = self.relu(self.f_3_r(x))
        x = x + x_r
        x_r = self.relu(self.f_4_r(x))
        x = x + x_r
        x = self.relu(self.f_5(x))

        x = torch.mean(x, dim=-2)  # [BS, x_dim, y_dim, h_size]

        # g network
        x = self.relu(self.g_1(x))
        x = self.relu(self.g_2(x))
        #
        x = torch.mean(x, dim=(-2, -3))  # [BS, h_size]

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


# Turn data from a table into all pairs of (x_i, y_i)
def process_batch(xs: torch.Tensor, ys: torch.Tensor):
    # output.shape = [xdim, ydim, num_samples, 2]
    # xs.shape, ys.shape = [num_samples, xdim or ydim]

    assert xs.shape[0] == ys.shape[0]

    xdim = xs.shape[1]
    ydim = ys.shape[1]
    num_samples = ys.shape[0]

    pair_output = torch.zeros([xdim, ydim, num_samples, 2])

    for k in range(num_samples):

        for i, x in enumerate(xs[k]):
            for j, y in enumerate(ys[k]):
                pair_output[i, j, k] = torch.tensor([x, y])

    return pair_output


def train_batch(model, data_batches: torch.Tensor, ds_labels: torch.Tensor, gamma=1):
    # data_batches.shape    = [BS, x_dim, y_dim, num_samples]
    # labels.shape          = [BS]
    # Reshape data into format

    bs = data_batches.shape[0]
    assert data_batches.shape[0] == ds_labels.shape[0]

    features = model(data_batches)  # [BS, out_dim]

    # Get every distinct combination of features, ignoring diagonals. Split into same ds and different ds.
    # Assume number of same and diff are both nonzero.

    same_1, same_2 = [], []
    diff_1, diff_2 = [], []
    for i in range(bs):
        for j in range(0, i, 1):
            # If features came from same class
            same = ds_labels[i] == ds_labels[j]
            if same:
                same_1.append(features[i])
                same_2.append(features[j])
            else:
                diff_1.append(features[i])
                diff_2.append(features[j])

    same_1, same_2 = torch.stack(same_1), torch.stack(same_2)
    diff_1, diff_2 = torch.stack(diff_1), torch.stack(diff_2)

    # Similarity, measured between 0 and 1
    same_sim = torch.norm(same_1 - same_2, dim=1)  # / (torch.norm(same_diff_1, dim=1) + torch.norm(same_diff_2, dim=1))
    diff_sim = torch.norm(diff_1 - diff_2, dim=1)  # / (torch.norm(diff_diff_1, dim=1) + torch.norm(diff_diff_2, dim=1))

    # Calculate loss
    # Original paper uses log of exp. Skip exp-log here and get loss directly.
    exp_same_sim = - gamma * same_sim
    loss_same = torch.mean(exp_same_sim)
    exp_diff_sim = torch.exp(- gamma * diff_sim)
    loss_diff = torch.mean(torch.log((1 - exp_diff_sim)))

    loss = -(loss_same + loss_diff)

    return loss


def train_loop():
    model = SetSetModel(h_size=64, out_size=64)

    optimiser = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    dl = ToyDataloader(bs=6, repeat_frac=1 / 2)

    for i in range(5000):
        for data, ds_label in dl:
            optimiser.zero_grad()
            loss = train_batch(model, data, ds_label)
            loss.backward()
            optimiser.step()

        if i % 10 == 0:
            print(loss.item())


if __name__ == "__main__":
    train_loop()
