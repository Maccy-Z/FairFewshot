import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np

np.random.seed(0)
random.seed(0)


# Mnist isn't sorted, so sort images by label.
class MnistMetaDataloader:
    def __init__(self, train, num_class, num_imgs, bs, epoch_len, num_train_class=7, device="cuda"):
        """

        :param train: train/test split
        :param num_class: Number of classes to sample from for each batch
        :param num_imgs: Images per class
        :param bs: Number batches to sample from each time
        :param epoch_len: Repeats before dataloader stops sending new iterations
        :param num_train_class: Number of digits to use in training, Rest are used for testing
        """

        self.num_class = num_class
        self.num_imgs = num_imgs
        self.epoch_len = epoch_len
        self.bs = bs

        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize((0.1307,), (0.3081,))  # Normalize image pixel values
        ])
        ds = datasets.MNIST(root='./data', train=train, transform=transform, download=True)

        numbers = np.random.choice(range(10), size=10, replace=False)
        train_nums, test_nums = numbers[:num_train_class], numbers[num_train_class:]

        self.train_nums, self.test_nums = train_nums, test_nums

        self.mnist_train = {label: [] for label in train_nums}
        self.mnist_test = {label: [] for label in test_nums}

        for img, label in ds:
            img = img.to(device=device)
            if label in train_nums:
                self.mnist_train[label].append(img)
            elif label in test_nums:
                self.mnist_test[label].append(img)
            else:
                exit(43)

    def __iter__(self):
        for _ in range(self.epoch_len):
            img_batchs, lab_batchs = [], []
            for _ in range(self.bs):
                # Which numbers to pick images from
                classes = np.random.choice(self.train_nums, self.num_class, replace=False)

                labels = np.repeat(classes, self.num_imgs)
                lab_batchs.append(torch.from_numpy(labels))

                img_batch = []
                # Sample from classes.
                for clas in classes:
                    wanted_classes = self.mnist_train[clas]
                    sample = random.sample(wanted_classes, self.num_imgs)

                    sample = torch.stack(sample)
                    img_batch.append(sample)

                img_batch = torch.stack(img_batch)
                img_batchs.append(img_batch)

            img_batchs = torch.stack(img_batchs)
            lab_batchs = torch.stack(lab_batchs)

            yield img_batchs, lab_batchs


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    test_ds = MnistMetaDataloader(num_class=3, num_imgs=3, epoch_len=5, train=False, bs=2, num_train_class=3, device="cpu")

    for imgs, labs in test_ds:
        print(imgs.shape, labs.shape)
