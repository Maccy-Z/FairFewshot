import os
import torch
from main import *
from AllDataloader import MyDataSet

BASEDIR = '/Users/kasiakobalczyk/FairFewshot'

def get_num_rows_cols():
    all_data_names = os.listdir(f'{BASEDIR}/datasets/data')
    all_data_names.remove('info.json')
    all_data_names.remove('.DS_Store')

    all_datasets = [
        MyDataSet(d, num_rows=0, num_targets=0, binarise=True, split="all") 
        for d in all_data_names]
    num_cols_ls = [d.ds_cols - 1 for d in all_datasets]
    num_cols_dict = dict(zip(all_data_names, num_cols_ls))
    num_rows_ls = [d.ds_rows for d in all_datasets]
    num_rows_dict = dict(zip(all_data_names, num_rows_ls))
    return num_rows_dict, num_cols_dict

def load_model(save_no):
    save_dir = f'{BASEDIR}/saves/save_{save_no}'
    state_dict = torch.load(f'{save_dir}/model.pt')
    cfg_all = get_config(cfg_file=f'{save_dir}/defaults.toml')
    cfg = cfg_all["DL_params"]
    model = ModelHolder(cfg_all=cfg_all)
    model.load_state_dict(state_dict['model_state_dict'])
    return model, cfg_all
    
def get_batch(dl, num_rows):
    xs, ys, model_id = next(iter(dl))
    xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
    ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
    xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
    ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
    ys_target = ys_target.view(-1)

    return model_id, xs_meta, xs_target, ys_meta, ys_target