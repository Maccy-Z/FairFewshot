import torch
from Fewshot.main import *
import os
import toml

def load_model(save_no):
    save_dir = os.path.join(f'./saves/save_{save_no}')
    model_save = torch.load(os.path.join(save_dir, 'model.pt'))
    all_cfgs = toml.load(os.path.join(save_dir, 'defaults.toml'))
    model = ModelHolder(cfg_all=all_cfgs)
    model.load_state_dict(model_save['model_state_dict'])
    return model, all_cfgs

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

def get_flat_embedding(model, xs_meta, ys_meta):
    pairs_meta = d2v_pairer(xs_meta, ys_meta)
    embed_meta, pos_enc = model.forward_meta(pairs_meta)
    return embed_meta, pos_enc