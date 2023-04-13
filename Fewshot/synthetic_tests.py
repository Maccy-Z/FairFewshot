import torch
from main import ModelHolder
from dataloader import d2v_pairer


def gen_synthetic(num_cols, num_rows, num_targets):
    xs_meta = (torch.arange(num_rows) - 5) / 5
    ys_meta = (xs_meta < 0.5).long()

    xs_meta = xs_meta.view(1, num_rows, num_cols)
    ys_meta = ys_meta.view(1, num_rows)

    xs_target = (torch.arange(num_targets) - 5) / 5
    ys_target = (xs_target > 0).long()

    xs_target = xs_target.view(1, num_targets, num_cols)
    ys_target = ys_target.view(1, num_targets)
    return xs_meta, ys_meta, xs_target, ys_target


def main():
    save_no = 27
    BASEDIR = '.'
    save_dir = f'{BASEDIR}/saves/save_{save_no}'

    state_dict = torch.load(f'{save_dir}/model.pt')
    model = ModelHolder(cfg_file=f"./saves/save_{save_no}/defaults.toml")
    model.load_state_dict(state_dict['model_state_dict'])

    xs_meta, ys_meta, xs_target, ys_target = gen_synthetic(num_cols=1, num_rows=10, num_targets=10)

    print(xs_meta.flatten().numpy(), ys_meta.numpy())
    print(xs_target.flatten().numpy(), ys_target.numpy())

    pairs_meta = d2v_pairer(xs_meta, ys_meta)
    embed_meta, pos_enc = model.forward_meta(pairs_meta)

    ys_pred_targ = model.forward_target(xs_target, embed_meta, pos_enc).view(-1, 2)
    predicted_labels = torch.argmax(ys_pred_targ, dim=1)

    print(predicted_labels.numpy())


if __name__ == "__main__":
    main()
