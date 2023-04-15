# Tests on 1D simple synthetic data

import torch
from main import ModelHolder
from dataloader import d2v_pairer
import matplotlib.pyplot as plt


def gen_synthetic(num_cols, num_rows, num_targets):
    def ys_fn(xs):
        return (xs > -0.).long()

    xs_meta = (torch.arange(num_rows) - 5) / 5
    ys_meta = ys_fn(xs_meta)

    xs_meta = xs_meta.view(1, num_rows, num_cols)
    ys_meta = ys_meta.view(1, num_rows)

    xs_target = (torch.arange(num_targets) - num_targets / 2) / 5
    ys_target = ys_fn(xs_target)

    xs_target = xs_target.view(1, num_targets, num_cols)
    ys_target = ys_target.view(1, num_targets)
    return xs_meta, ys_meta, xs_target, ys_target


def main():
    save_no = 1
    BASEDIR = '.'
    save_dir = f'{BASEDIR}/saves/save_{save_no}'

    state_dict = torch.load(f'{save_dir}/model.pt')
    model = ModelHolder(cfg_file=f"./saves/save_{save_no}/defaults.toml")
    model.load_state_dict(state_dict['model_state_dict'])

    xs_meta, ys_meta, xs_target, ys_target = gen_synthetic(num_cols=1, num_rows=10, num_targets=60)

    print(xs_meta.flatten().numpy(), ys_meta.numpy())
    print(xs_target.flatten().numpy(), ys_target.numpy())

    pairs_meta = d2v_pairer(xs_meta, ys_meta)
    embed_meta, pos_enc = model.forward_meta(pairs_meta)

    ys_pred_targ = model.forward_target(xs_target, embed_meta, pos_enc).view(-1, 2)
    predicted_labels = torch.argmax(ys_pred_targ, dim=1)

    print(xs_meta.shape, ys_meta.shape)
    plt.scatter(xs_meta.flatten(), ys_meta.flatten(), label="Meta")
    plt.plot(xs_target.flatten(), predicted_labels.flatten(), label="Target", c="y")
    plt.legend()
    plt.show()

    print(ys_pred_targ.detach().numpy())
    print(predicted_labels.numpy())


if __name__ == "__main__":
    main()
