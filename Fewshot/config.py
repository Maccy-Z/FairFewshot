import toml


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
                             "model_load": "model_main",  # Which D2V to load from

                             "weight_hid_dim": 64,      # Weight generator hidden dimension
                             "gen_layers": 2,           # Weight deocder layers

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

                 "Optim": {"lr": 3e-4},

                 "DL_params": {"bs": 3,
                               "num_rows": 5,
                               "num_targets": 5,
                               "ds_group": -1,          # Group of datasets from which to select from. -1 for full dataset
                               "balance_train": True,   # Balance dataloader during training
                               "one_v_all": True,       # How to binarise during training
                               "num_cols": None,
                               "train_data_names": ['statlog-heart', 'horse-colic', 'fertility', 'post-operative'],
                               "val_data_names": ['breast-cancer'],
                               "shuffle_cols": False,
                               "miss_rate": 0.8,
                               "fixed_num_cols" : False
                               },

                 "Settings": {"num_epochs": 100,      # Number of trainin epochs
                              "val_duration": 200,      # Number of batches of validation
                              "val_interval": 1000,     # Number of batches to train for each epoch
                              "dataset": "dummy",
                              },
                 }

    save_dict["NN_dims"]["gat_in_dim"] = save_dict["NN_dims"]["pos_enc_dim"] + 1
    with open("./Fewshot/defaults.toml", "w") as f:
        toml.dump(save_dict, f)


if __name__ == "__main__":
    print("Resetting config to defaults")
    write_toml()
    get_config()
