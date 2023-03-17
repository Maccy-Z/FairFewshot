import torch
import torch.nn as nn
from Mnist import MnistMetaDataloader

torch.manual_seed(0)


class CNN(nn.Module):
    def __init__(self, out_dim=32):
        super(CNN, self).__init__()
        self.out_dim = out_dim
        # Input shape [bs, set_size, 1, height, width]
        # Need to flatten out to [bs*set_size, 1, h, w] then unroll for equivariant layer.

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        # x.shape                   # [bs, num_class, img_per_class, 1, h, w]
        bs = x.shape[0]
        num_class = x.shape[1]
        x = x.view(-1, 1, 28, 28)  # [bs*set_size, 1, h, w]
        # Conv layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # [bs*set_size, out_dim]

        x = x.view(bs, num_class, -1, self.out_dim)
        return x


# Prototype network probabilies from predicted features. Take last feature from each class as prediction and rest as conditioning.
def split_features(features):
    # features.shape # [BS, num_classes, img_per_class, out_dim]
    num_class = features.shape[1]

    test = features[:, :, -1]
    prototypes = features[:, :, :-1]
    prototypes = torch.mean(prototypes, dim=2)

    test = torch.arange(6).view(1, 3, 2).float()
    prototypes = torch.tensor([[[0., 1], [3, 4], [10, 10]]])

    # Find norm between every pair by tiling one and interleaving the other.
    # tile: [1, 2, 3 ,1, 2, 3]
    # repeat: [1, 1, 2, 2, 3, 3]
    test = torch.repeat_interleave(test, num_class, dim=-2)
    prototypes = torch.tile(prototypes, (num_class, 1))

    distances = -torch.norm(test - prototypes, dim=-1)


    # Find true distance on diagonal of matrix.
    distances = distances.view(-1, num_class, num_class)    # [bs, test no, prototype no]

    # Calculates every probabity. Only need diagonal
    probs = torch.nn.Softmax(dim=-1)(distances)
    prob_correct = torch.diagonal(probs, dim1=1, dim2=2)


    print(prob_correct)


def main():
    num_class = 3
    dl = MnistMetaDataloader(num_class=num_class, num_imgs=6, epoch_len=5, train=False, bs=2, device="cpu")

    model = CNN(out_dim=1)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)

    for imgs, labels in dl:
        # imgs.shape = [BS, num_classes * img_per_class, 1, 28, 28]
        # Find features of every image and split out for each class

        features = model(imgs)
        split_features(features)

        exit(3)


if __name__ == "__main__":
    main()
