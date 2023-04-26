import toml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ds-group', type=int)   
args = parser.parse_args()

def get_config(cfg_file=None):
    if cfg_file is None:
        cfg_file = "./Fewshot/defaults.toml"

    with open(cfg_file, "r") as f:
        config = toml.load(f)
    return config


def write_toml():
    save_dict = {"NN_dims": {"set_h_dim": 64,           # D2v hidden dimension
                             "set_out_dim": 64,         # D2v output dimension
                             "d2v_layers": [3, 2, 3],   # layers of d2v. first 3 are d2v dims, last positional encoder
                             "pos_depth": 2,            # Depth of positional encoder.
                             "pos_enc_dim": 15,          # Dimension of the positional encoder output
                             "load_d2v": False,          # Load pretrained datset2vec
                             "freeze_d2v": False,       # Continue training datset2vec
                             "model_load": "model_3",  # Which D2V to load from

                             "weight_hid_dim": 64,      # Weight generator hidden dimension
                             "gen_layers": 1,           # Weight deocder layers

                             "gat_heads": 2,            # Number of heads in GAT
                             "gat_layers": 2,           # Depth of GAT
                             "gat_hid_dim": 128,        # Hidden dimension of GAT
                             "gat_out_dim": 16,         # Output of GAT

                             "reparam_weight": False,   # Reparametrise outputs
                             "reparam_pos_enc": False,

                             "weight_bias": "off",     # Zero: zero init. # Off, disable bias completely # Anything else: default init
                             "pos_enc_bias": "zero",

                             "norm_lin": True,         # Normalise weights by dividing by L2 norm. final classification weight
                             "norm_weights": True,      # GAT weights
                             "learn_norm": True,
                             },

                 "Optim": {"lr": 5e-4,
                           "eps": 3e-4},

                 "DL_params": {"bs": 5,
                               "num_rows": 5,
                               "num_targets": 5,
                               "ds_group": args.ds_group,   # Group of datasets from which to select from. -1 for full dataset
                               "binarise" : True,
                               "decrease_col_prob": 0.12,
                               "split_file" : 'my_splits',
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
