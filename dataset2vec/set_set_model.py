import torch
import torch.nn as nn

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

#
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


def train_batch(model, data_batches, labels, gamma=1):
    # data_batches.shape = [2, x_dim, y_dim, num_samples]
    # Reshape data into format

    features = model(data_batches)

    feature_diff = features[0] - features[1]
    i = torch.exp(- gamma * torch.norm(feature_diff))
    print(i)

    return features


if __name__ == "__main__":
    m = SetSetModel(h_size=64, out_size=64)

    a = torch.zeros([2, 4, 1, 10, 2])
    a[1] = torch.ones_like(a[1])

    b = train_batch(m, a, torch.tensor([1]))

