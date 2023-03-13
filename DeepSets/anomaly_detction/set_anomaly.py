# Set anomaly detection on Mnist, using CNN and equivariant layer
import torch
import torch.nn as nn
import torch.optim as optim

from DeepSets.anomaly_detction.mnist_anomaly_dl import MnistAnomaly
from DeepSets.anomaly_detction.equivariant_layers import EquivariantLayer

import random

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input shape [bs, set_size, 1, height, width]
        # Need to flatten out to [bs*set_size, 1, h, w] then unroll for equivariant layer.

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # self.fc2 = nn.Linear(128, 128)

        # equivariant layers
        self.eq1 = EquivariantLayer(in_channel=128, out_channel=128)
        self.eq2 = EquivariantLayer(in_channel=128, out_channel=128)
        self.eq3 = EquivariantLayer(in_channel=128, out_channel=1)

    def forward(self, x):
        bs = x.shape[0]
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
        # x = self.fc2(x)             # [bs*set_size, 128]

        # Equivariant layers
        x = x.view(bs, -1, 128)

        x = self.eq1(x)
        x = self.relu(x)
        # x = self.eq2(x)
        # x = self.relu(x)
        x = self.eq3(x)
        x = x.squeeze(-1)  # [bs, set_size]
        return x


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    # Set device to GPU if available, else CPU
    device = torch.device('cuda')

    SET_SIZE = 5  # Excluding odd item out
    TEST_SIZE = 5
    BS = 8

    train_ds = MnistAnomaly(train=True, num_imgs=SET_SIZE, bs=BS, epoch_len=5000)
    test_ds = MnistAnomaly(train=False, num_imgs=TEST_SIZE, bs=BS, epoch_len=5000)

    # Instantiate model and move to device
    model = CNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    labels = [SET_SIZE for _ in range(BS)]  # Odd image out is always last

    labels = torch.tensor(labels, device=device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    print("Training Model")

    # Train model
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, images_batch in enumerate(train_ds):

            optimizer.zero_grad()

            outputs = model(images_batch)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 500 == 0:
                print()
                print(outputs[0])
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    print()
    print()
    print("Training Completed, saving model")
    with open("../save_model", "wb") as f:
        torch.save(model, f)

    print("Model Saved, evaluating model")

    # Test model
    model.eval()
    true_labels = torch.ones(BS, device=device) * TEST_SIZE
    with torch.no_grad():
        correct = 0
        total = 0
        for images in test_ds:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == true_labels).sum().item()
            total += BS

        accuracy = 100 * correct / total

        print(f'{accuracy = :.4g}%')
