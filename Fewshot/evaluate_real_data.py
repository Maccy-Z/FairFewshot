import torch
from main import *
from dataloader import d2v_pairer
from Fewshot.AllDataloader import AllDatasetDataLoader
import os
import toml


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class ZeroModel:
    def fit(self, X, y):

        self.y = torch.mode(y)[0]
        return

    def predict(self, X):

        return np.ones(X.shape[0]) * self.y.numpy()


def get_batch(dl, num_rows):
    try:
        xs, ys, model_id = next(iter(dl))
    except:
        xs, ys = next(iter(dl))
        model_id = []
    xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
    ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
    xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
    ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
    ys_target = ys_target.view(-1)

    if len(model_id) > 0:
        return model_id, xs_meta, xs_target, ys_meta, ys_target
    return xs_meta, xs_target, ys_meta, ys_target

def get_embedding(xs_meta, ys_meta, model):
    pairs_meta = d2v_pairer(xs_meta, ys_meta)
    embed_meta, pos_enc = model.forward_meta(pairs_meta)
    return embed_meta, pos_enc


def get_predictions(xs_meta, xs_target, ys_meta, model):
    # if ys_meta.min() == ys_meta.max():
    # predictions = torch.ones(xs_target.shape[1])# * ys_meta[0]
    # print()
    # print(predictions.shape)

    embed_meta, pos_enc = get_embedding(xs_meta, ys_meta, model)
    ys_pred_target = model.forward_target(xs_target, embed_meta, pos_enc)
    return ys_pred_target


def get_accuracy(ys_pred_target, ys_target):
    ys_pred_target_labels = torch.argmax(ys_pred_target.view(-1, 2), dim=1)
    accuracy = (ys_pred_target_labels == ys_target).sum().item() / len(ys_target)
    return accuracy

def get_baseline_predictions(model, xs_meta, ys_meta, xs_target, i):
    ys_meta = ys_meta.detach()[i].flatten()
    xs_meta = xs_meta.detach()[i]
    xs_target = xs_target.detach()[i]
    # print(ys_meta)
    if ys_meta.min() == ys_meta.max():
        predictions = np.ones(xs_target.shape[0]) * ys_meta[0].numpy()

    else:
        model.fit(X=xs_meta, y=ys_meta)
        predictions = model.predict(X=xs_target)
    return predictions


def get_baseline_accuracy(model, bs, xs_meta, ys_meta, xs_target, ys_target):
    ys_lr_target_labels = [get_baseline_predictions(model,
                                                    xs_meta,
                                                    ys_meta,
                                                    xs_target,
                                                    i) for i in range(bs)]

    ys_lr_target_labels = np.array(ys_lr_target_labels).flatten()
    accuracy = (ys_lr_target_labels == np.array(ys_target)).sum().item() / len(ys_target)
    return accuracy


def main():
    save_no = 20
    BASEDIR = '/mnt/storage_ssd/FairFewshot'
    save_dir = os.path.join(BASEDIR, f'saves/save_{save_no}')

    state_dict = torch.load(os.path.join(save_dir, 'model.pt'))
    model = ModelHolder()
    model.load_state_dict(state_dict['model_state_dict'])

    cfg = toml.load(os.path.join(save_dir, 'defaults.toml'))["DL_params"]

    num_rows = cfg["num_rows"]
    num_targets = cfg["num_targets"]

    bs = 1
    baseline_models = [LogisticRegression(max_iter=1000), SVC(), ZeroModel()]
    baseline_model_names = ['LR', 'SVC', "Baseline model"]

    for num_cols in range(1, 10):
        acc = []
        baseline_acc = {name: [] for name in baseline_model_names}
        val_dl = AllDatasetDataLoader(bs=bs, num_rows=num_rows, num_targets=num_targets,
                                      num_cols=num_cols, split="val")

        for j in range(600):
            # Fewshot predictions
            xs_meta, xs_target, ys_meta, ys_target = get_batch(val_dl, num_rows)
            ys_pred_target = get_predictions(xs_meta=xs_meta, xs_target=xs_target, ys_meta=ys_meta, model=model)
            acc.append(get_accuracy(ys_pred_target, ys_target))

            # Predictions for baseline models
            for base_model, model_name in zip(baseline_models, baseline_model_names):
                baseline_acc[model_name].append(get_baseline_accuracy(
                    model=base_model,
                    bs=bs,
                    xs_meta=xs_meta,
                    xs_target=xs_target,
                    ys_meta=ys_meta,
                    ys_target=ys_target
                ))
        print('---------------------')
        print(f'num_cols: {num_cols}')
        print(f'Fewshot mean acc: {np.mean(acc):.3f}')
        for model_name in baseline_model_names:
            print(f'{model_name} mean acc: {np.mean(baseline_acc[model_name]):.3f}')


if __name__ == "__main__":

    main()
