import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random


# Mnist isn't sorted, so sort images by label.
class MnistAnomaly:
    def __init__(self, train, num_imgs, bs, epoch_len, device="cuda"):
        self.num_imgs = num_imgs
        self.epoch_len = epoch_len
        self.bs = bs

        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize((0.1307,), (0.3081,))  # Normalize image pixel values
        ])
        ds = datasets.MNIST(root='./data', train=train, transform=transform, download=True)
        self.mnist = {label: [] for label in range(10)}
        for img, label in ds:
            img = img.to(device=device)
            self.mnist[label].append(img)

    def __iter__(self):
        numbers = list(range(10))

        for _ in range(self.epoch_len):
            img_batch = []
            for _ in range(self.bs):
                main_class, second_class = random.sample(numbers, 2)

                main_imgs = random.sample(self.mnist[main_class], self.num_imgs)
                second_img = random.sample(self.mnist[second_class], 1)

                img_group = main_imgs + second_img

                img_group = torch.stack(img_group)

                img_batch.append(img_group)

            yield torch.stack(img_batch)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # train_ds = MnistAnomaly(num_imgs=3, epoch_len=5, train=True, bs=1)
    test_ds = MnistAnomaly(num_imgs=3, epoch_len=5, train=False, bs=1, device="cpu")

    for img_epoch in test_ds:
        for batch in img_epoch:
            for img in batch:
                plt.imshow(img[0])
                plt.show()


