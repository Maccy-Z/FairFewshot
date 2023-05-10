# Tests on 2D simple synthetic data
import sys
sys.path.append("/mnt/storage_ssd/FairFewshot/Fewshot")
import torch
from main import ModelHolder
from dataloader import d2v_pairer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from config import get_config


save_no = 41


def ys_fn(point):
    x, y, = point

    label = (x ** 2 + y ** 2) < 100
    label = x + y > -5000.0
    return label


def gen_synthetic(point):
    # Train/meta data
    # xx, yy = torch.meshgrid(torch.linspace(-1, 1, 4),
    #                         torch.linspace(-1, 1, 4))
    # xs_meta = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    xs_meta = torch.tensor([point])

    #xs_meta += 0.1 * torch.randn_like(xs_meta)
    # xs_meta = torch.tensor([[0., 0.]])

    ys_meta = torch.stack([ys_fn(point) for point in xs_meta]).long()
    xs_meta = xs_meta.view(1, -1, 2)
    ys_meta = ys_meta.view(1, -1)

    # Test/target data
    xx, yy = torch.meshgrid(torch.linspace(-2, 2, 100),
                            torch.linspace(-2, 2, 100))
    xs_target = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    # Remove original point
    # index = torch.all(xs_target == torch.tensor(point), dim=1).nonzero()
    # xs_target = torch.cat((xs_target[:index], xs_target[index + 1:]))

    xs_target = xs_target.view(1, -1, 2)

    return xs_meta, ys_meta, xs_target, xx, yy


def model_predictions(xs_meta, ys_meta, xs_target):
    BASEDIR = '.'
    save_dir = f'{BASEDIR}/saves/save_{save_no}'

    state_dict = torch.load(f'{save_dir}/model.pt')
    model = ModelHolder(cfg_all=get_config(f"./saves/save_{save_no}/defaults.toml"))
    model.load_state_dict(state_dict['model_state_dict'])

    pairs_meta = d2v_pairer(xs_meta, ys_meta)
    embed_meta, pos_enc = model.forward_meta(pairs_meta)
    ys_pred_target = model.forward_target(xs_target, embed_meta, pos_enc).squeeze()
    predicted_labels = torch.argmax(ys_pred_target, dim=1)
    return predicted_labels

def maml_predcitions(xs_meta, ys_meta, xs_target):

    BASEDIR = '.'
    save_dir = f'{BASEDIR}/saves/save_{save_no}'

    state_dict = torch.load(f'{save_dir}/model.pt')
    model = ModelHolder(cfg_all=get_config(f"./saves/save_{save_no}/defaults.toml"))
    model.load_state_dict(state_dict['model_state_dict'])

    pairs_meta = d2v_pairer(xs_meta, ys_meta)
    with torch.no_grad():
        embed_meta, pos_enc = model.forward_meta(pairs_meta)

    embed_meta.requires_grad = True
    pos_enc.requires_grad = True
    optim_pos = torch.optim.Adam([pos_enc], lr=0.001)
    optim_embed = torch.optim.Adam([embed_meta], lr=0.075)
    for _ in range(15):
        # Make predictions on meta set and calc loss
        preds = model.forward_target(xs_meta, embed_meta, pos_enc)
        loss = torch.nn.functional.cross_entropy(preds.squeeze(), ys_meta.long().squeeze())
        loss.backward()
        optim_pos.step()
        optim_embed.step()
        optim_embed.zero_grad()
        optim_pos.zero_grad()

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
    # point = [1, 0]
    # xs_meta, ys_meta, xs_target, xx, yy = gen_synthetic(point)
    #
    # print(xs_meta.shape, ys_meta.shape, xs_target.shape)
    # model_preds = model_predictions(xs_meta=xs_meta, ys_meta=ys_meta, xs_target=xs_target)
    #
    # target = [-point[0], -point[1]]
    # # One shot
    # plt.subplot(1, 2, 1)
    # plt.title(f'meta: {point}, target: {target}', fontsize=15)
    # plt.scatter(xs_meta[0, :, 0], xs_meta[0, :, 1], c=ys_meta.squeeze(), cmap='bwr',s=200)
    # plt.scatter(-point[0], -point[1], c="r", marker="x",s=200)
    #
    # plt.contourf(xx, yy, 1 - model_preds.reshape(xx.shape), alpha=0.2, cmap='bwr')
    # plt.xticks([-2, 0, 2], [-2, 0, 2], fontsize=15)
    # plt.yticks([-2, 0, 2], [-2, 0, 2], fontsize=15)

    xs_meta = torch.tensor([[[0, 0, 0, 0, -1]]])
    ys_meta = torch.tensor([[1]])
    xs_target = torch.tensor([[[0, 0, 0, 0, 1], [1., 1., 1, 0, 0]]])
    print(xs_meta.shape, ys_meta.shape, xs_target.shape)

    model_preds = model_predictions(xs_meta=xs_meta, ys_meta=ys_meta, xs_target=xs_target)

    print(model_preds)
    # Our model
    # plt.subplot(1, 2, 1)
    # plt.title("FLAT", fontsize=15)
    #
    # plt.scatter(xs_meta[0, :, 0], xs_meta[0, :, 1], c=ys_meta.squeeze(), cmap='bwr')
    # plt.contourf(xx, yy, model_preds.reshape(xx.shape), alpha=0.2, cmap='bwr')
    # plt.xticks([-2, 0, 2], [-2, 0, 2], fontsize=15)
    # plt.yticks([-2, 0, 2], [-2, 0, 2], fontsize=15)

    # plt.subplot(1, 2, 2)
    # plt.title("FLAT-MAML", fontsize=15)
    #
    # maml_preds = maml_predcitions(xs_meta=xs_meta, ys_meta=ys_meta, xs_target=xs_target)
    # plt.scatter(xs_meta[0, :, 0], xs_meta[0, :, 1], c=ys_meta.squeeze(), cmap='bwr')
    # plt.contourf(xx, yy, maml_preds.reshape(xx.shape), alpha=0.2, cmap='bwr')
    # plt.xticks([-2, 0, 2], [-2, 0, 2], fontsize=15)
    # plt.yticks([-2, 0, 2], [-2, 0, 2], fontsize=15)


    # # Logistic regression
    # lin_preds = sklearn_pred(xs_meta=xs_meta, ys_meta=ys_meta, xs_target=xs_target, model=LogisticRegression(max_iter=1000))
    #
    # plt.subplot(1, 3, 2)
    # plt.scatter(xs_meta[0, :, 0], xs_meta[0, :, 1], c=ys_meta.squeeze(), cmap='bwr', label="Meta")
    # plt.contourf(xx, yy, lin_preds.reshape(xx.shape), alpha=0.2, cmap='bwr')
    # plt.legend()
    # plt.title("Logistic Regression predictions")
    #
    # # SVC
    # lin_preds = sklearn_pred(xs_meta=xs_meta, ys_meta=ys_meta, xs_target=xs_target, model=SVC())
    #
    # plt.subplot(1, 3, 3)
    # plt.scatter(xs_meta[0, :, 0], xs_meta[0, :, 1], c=ys_meta.squeeze(), cmap='bwr', label="Meta")
    # plt.contourf(xx, yy, lin_preds.reshape(xx.shape), alpha=0.2, cmap='bwr')
    # plt.legend()
    # plt.title("SVC predictions")

    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)
    for _ in range(1):
        main()
