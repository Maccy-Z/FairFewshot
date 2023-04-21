from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import torch
from matplotlib import pyplot as plt


def ys_fn(point):
    x, y, = point
    label =x + y < 0.5
    return label


def gen_synthetic():
    # Train/meta data
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, 3),
                            torch.linspace(-1, 1, 4))
    xs_meta = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    xs_meta += 0.1 * torch.randn_like(xs_meta)
    ys_meta = torch.stack([ys_fn(point) for point in xs_meta]).long()
    xs_meta = xs_meta.view(1, -1, 2)
    ys_meta = ys_meta.view(1, -1)

    # Test/target data
    xx, yy = torch.meshgrid(torch.linspace(-1.3, 1.3, 100),
                            torch.linspace(-1.3, 1.3, 100))

    xs_target = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    # ys_target = [ys_fn(point) for point in xs_target]
    xs_target = xs_target.view(1, -1, 2)
    return xs_meta, ys_meta, xs_target, xx, yy


def main():
    xs_meta, ys_meta, xs_target, xx, yy = gen_synthetic()
    # print(xs_meta.shape, ys_meta.shape)
    clf = TabNetClassifier(device_name="cpu")

    print(f'{xs_meta = }')
    clf.fit(xs_meta.squeeze().numpy(), ys_meta.squeeze().numpy(),
            # eval_set=[(xs_meta.squeeze().numpy(), ys_meta.squeeze().numpy())], eval_name=["accuracy"],
            batch_size=8, virtual_batch_size=8, patience=50, drop_last=False)

    preds = clf.predict(xs_target.squeeze().numpy())

    print(preds)

    plt.scatter(xs_meta[0, :, 0], xs_meta[0, :, 1], c=ys_meta.squeeze(), cmap='bwr', label="Meta")
    plt.contourf(xx, yy, preds.reshape(xx.shape), alpha=0.2, cmap='bwr')
    plt.legend()
    plt.title("Tabnet predictions")
    plt.show()


if __name__ == "__main__":
    main()
