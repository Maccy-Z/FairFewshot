import torch
from DeepSets.anomaly_detction.mnist_anomaly_dl import MnistAnomaly
import random

random.seed(0)
torch.manual_seed(0)

device = "cuda"

with open("../save_model", "rb") as f:
    model = torch.load(f)
    model.eval()

BS = 8

print("Train set size = 5")

for size in range(2, 15):

    test_ds = MnistAnomaly(train=False, num_imgs=size, bs=BS, epoch_len=5000)

    # Test model
    true_labels = torch.ones(BS, device=device) * size
    with torch.no_grad():
        correct = 0
        total = 0
        for i, images in enumerate(test_ds):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == true_labels).sum().item()
            total += BS

        accuracy = 100 * correct / total

        print(f'test {size = }, {accuracy = :.4g}%')
