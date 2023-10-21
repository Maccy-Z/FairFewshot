import toml


def get_config(cfg_file=None):
    if cfg_file is None:
        cfg_file = "./Fewshot/configs.toml"

    with open(cfg_file, "r") as f:
        config = toml.load(f)
    return config


def write_toml():
    save_dict = {"NN_dims": {"set_h_dim": 64,  # D2v hidden dimension
                             "set_out_dim": 64,  # D2v output dimension
                             "d2v_layers": [4, 2, 4],  # layers of d2v.
                             "pos_depth": 2,  # Depth of positional encoder.
                             "pos_enc_dim": 15,  # Dimension of the positional encoder output
                             "model_load": "model_3",  # Which D2V to load from

                             "weight_hid_dim": 64,  # Weight generator hidden dimension
                             "gen_layers": 1,  # Weight deocder layers

                             "gat_heads": 2,  # Number of heads in GAT
                             "gat_layers": 2,  # Depth of GAT
                             "gat_hid_dim": 128,  # Hidden dimension of GAT
                             "gat_out_dim": 16,  # Output of GAT

                             "weight_bias": "off",  # Zero: zero init. # Off, disable bias completely # Anything else: default init
                             "pos_enc_bias": "zero",

                             "norm_lin": True,  # Normalise weights by dividing by L2 norm. final classification weight
                             "norm_weights": True,  # GAT weights
                             "learn_norm": True,
                             },

                 "Optim": {"lr": 5e-4,
                           "eps": 3e-4,
                           "decay": 1e-4},

                 "DL_params": {"bs": 3,
                               "num_rows": 10,
                               "num_targets": 15,
                               "ds_group": [0, -1],  # Group of datasets from which to select from. -1 for full dataset
                               "binarise": True,
                               "num_1s": None,
                               "num_cols": {'train': -2, 'val': -2},
                               "split_file": 'splits',
                               },

                 "Settings": {"num_epochs": 31,  # Number of train epochs
                              "val_duration": 100,  # Number of batches of validation
                              "val_interval": 2000,  # Number of batches to train for each epoch
                              "dataset": 'total',
                              },
                 }

    save_dict["NN_dims"]["gat_in_dim"] = save_dict["NN_dims"]["pos_enc_dim"] + 1
    with open("./Fewshot/configs.toml", "w") as f:
        toml.dump(save_dict, f)


if __name__ == "__main__":
    print("Setting up config file")
    write_toml()