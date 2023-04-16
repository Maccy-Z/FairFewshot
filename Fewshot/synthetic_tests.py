# Tests on 2D simple synthetic data
import torch
from main import ModelHolder
from dataloader import d2v_pairer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def ys_fn(point):
    x, y, = point

    label = (x**2 + y ** 2) > 0.8
    label = x + y < 0.5
    return label


def gen_synthetic():
    # Train/meta data
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, 3),
                            torch.linspace(-1, 1, 3))
    xs_meta = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    xs_meta += 0.1 * torch.randn_like(xs_meta)
    ys_meta = torch.stack([ys_fn(point) for point in xs_meta]).long()
    xs_meta = xs_meta.view(1, -1, 2)
    ys_meta = ys_meta.view(1, -1)

    # Test/target data
    xx, yy = torch.meshgrid(torch.linspace(-1.5, 1.5, 100),
                            torch.linspace(-1.5, 1.5, 100))

    xs_target = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    # ys_target = [ys_fn(point) for point in xs_target]
    xs_target = xs_target.view(1, -1, 2)
    return xs_meta, ys_meta, xs_target, xx, yy


def model_predictions(xs_meta, ys_meta, xs_target):
    save_no = 13
    BASEDIR = '.'
    save_dir = f'{BASEDIR}/saves/save_{save_no}'

    state_dict = torch.load(f'{save_dir}/model.pt')
    model = ModelHolder(cfg_file=f"./saves/save_{save_no}/defaults.toml")
    model.load_state_dict(state_dict['model_state_dict'])

    pairs_meta = d2v_pairer(xs_meta, ys_meta)
    embed_meta, pos_enc = model.forward_meta(pairs_meta)
    ys_pred_target = model.forward_target(xs_target, embed_meta, pos_enc).squeeze()
    predicted_labels = torch.argmax(ys_pred_target, dim=1)
    return predicted_labels


def sklearn_pred(xs_meta, ys_meta, xs_target, model):
    xs_meta, ys_meta = xs_meta.squeeze(), ys_meta.squeeze()
    xs_target = xs_target.squeeze()
    model.fit(X=xs_meta, y=ys_meta)
    predictions = model.predict(X=xs_target)
    return predictions

def main():
    xs_meta, ys_meta, xs_target, xx, yy = gen_synthetic()

    # Our model
    model_preds = model_predictions(xs_meta=xs_meta, ys_meta=ys_meta, xs_target=xs_target)

    plt.subplot(1, 3, 1)
    plt.scatter(xs_meta[0, :, 0], xs_meta[0, :, 1], c=ys_meta.squeeze(), cmap='bwr', label="Meta")
    plt.contourf(xx, yy, model_preds.reshape(xx.shape), alpha=0.2, cmap='bwr')
    plt.legend()
    plt.title("Model predictions")


    # Logistic regression
    lin_preds = sklearn_pred(xs_meta=xs_meta, ys_meta=ys_meta, xs_target=xs_target, model=LogisticRegression(max_iter=1000))

    plt.subplot(1, 3, 2)
    plt.scatter(xs_meta[0, :, 0], xs_meta[0, :, 1], c=ys_meta.squeeze(), cmap='bwr', label="Meta")
    plt.contourf(xx, yy, lin_preds.reshape(xx.shape), alpha=0.2, cmap='bwr')
    plt.legend()
    plt.title("Logistic Regression predictions")

    # SVC
    lin_preds = sklearn_pred(xs_meta=xs_meta, ys_meta=ys_meta, xs_target=xs_target, model=SVC())

    plt.subplot(1, 3, 3)
    plt.scatter(xs_meta[0, :, 0], xs_meta[0, :, 1], c=ys_meta.squeeze(), cmap='bwr', label="Meta")
    plt.contourf(xx, yy, lin_preds.reshape(xx.shape), alpha=0.2, cmap='bwr')
    plt.legend()
    plt.title("SVC predictions")

    fig = plt.gcf()
    fig.set_size_inches(15, 5)
    plt.show()


if __name__ == "__main__":
    # torch.manual_seed(0)
    # import numpy as np
    # np.random.seed(0)
    main()
