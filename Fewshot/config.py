import torch.random

import toml
import dataclasses
import numpy as np

@dataclasses.dataclass
class Config:
    # Dataloader params
    min_row_per_label: int = 15     # Minimum number of rows in dataset
    min_cols: int = 5               # Minimum number of dataset columns

    fix_per_label: bool = False      # Fix N_meta per label instead of total
    N_meta: int = 10                 # N rows in meta
    N_target: int = 10               # N rows in target

    col_fmt: str = 'uniform'        # How to sample number of columns per batch
    normalise: bool = True          # Normalise predictors
    # sample_fmt: str = 'balanced'    # Sample rows randomly or balance labels

    DS_DIR: str = './datasets'
    ds_group: str = '0'             # Datasets to sample from. List or filename

    # Model parameters
    proto_dim: int = 16

    # RNGs
    seed: int = 0

    def __post_init__(self):
        #assert self.min_row_per_label >= self.N_meta + self.N_target

        self.RNG = np.random.default_rng(seed=self.seed)
        self.T_RNG = torch.Generator()
        self.T_RNG.manual_seed(self.seed)


def get_config(cfg_file=None):
    if cfg_file is None:
        cfg_file = "./Fewshot/defaults.toml"

    with open(cfg_file, "r") as f:
        config = toml.load(f)
    return config


def write_toml():
    save_dict = {"NN_dims": {"set_h_dim": 64,           # D2v hidden dimension
                             "set_out_dim": 64,         # D2v output dimension
                             "d2v_layers": [4, 2, 4],   # layers of d2v. first 3 are d2v dims, last positional encoder
                             "pos_depth": 2,            # Depth of positional encoder.
                             "pos_enc_dim": 15,          # Dimension of the positional encoder output

                             "weight_hid_dim": 64,      # Weight generator hidden dimension
                             "gen_layers": 1,           # Weight deocder layers

                             "gat_heads": 2,            # Number of heads in GAT
                             "gat_layers": 2,           # Depth of GAT
                             "gat_hid_dim": 128,        # Hidden dimension of GAT
                             "gat_out_dim": 16,         # Output of GAT


                             "weight_bias": "off",     # Zero: zero init. # Off, disable bias completely # Anything else: default init
                             "pos_enc_bias": "zero",

                             "norm_lin": True,         # Normalise weights by dividing by L2 norm. final classification weight
                             "norm_weights": True,      # GAT weights
                             "learn_norm": True,
                             },

                 "Optim": {"lr": 5e-4,
                           "eps": 3e-4,
                           "decay": 1e-4},

                 "DL_params": {"bs": 3,
                               "num_meta": 10,
                               "num_targets": 10,
                               "ds_group": 0,          # Group of datasets from which to select from. -1 for full dataset
                               "binarise" : True,
                               "num_1s": None,
                               "num_cols": {'train': -2, 'val': -2},
                               "split_file" : 'med_splits_2',
                               },

                 "Settings": {"num_epochs": 51,      # Number of train epochs
                              "val_duration": 100,      # Number of batches of validation
                              "val_interval": 2000,     # Number of batches to train for each epoch
                              "dataset": "my_split",
                              },
                 }

    save_dict["NN_dims"]["gat_in_dim"] = save_dict["NN_dims"]["pos_enc_dim"] + 1
    with open("./Fewshot/defaults.toml", "w") as f:
        toml.dump(save_dict, f)


if __name__ == "__main__":
    print("Resetting config to defaults")
    write_toml()
    cfg = get_config()
    for k, v in cfg.items():
        print(k)
        print(v)
