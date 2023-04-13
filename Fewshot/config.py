import toml


def get_config(cfg_file=None):
    if cfg_file is None:
        cfg_file = "./Fewshot/defaults.toml"

    with open(cfg_file, "r") as f:
        config = toml.load(f)
    return config


def write_toml():
    save_dict = {"NN_dims": {"pos_enc_dim": 15,         # Dimension of the positional encoder output
                             "set_h_dim": 64,           # D2v hidden dimension
                             "set_out_dim": 32,         # D2v output dimension
                             "weight_hid_dim": 64,      # Weight generator hidden dimension
                             "gat_heads": 2,            # Number of heads in GAT
                             "gat_hid_dim": 128,        # Hidden dimension of GAT
                             "gat_out_dim": 16,         # Output of GAT
                             "d2v_layers": [3, 2, 3, 2],# layers of d2v. first 3 are d2v dims, last positional encoder
                             "gen_layers": 2,           # Weight deocder layers
                             "gat_layers": 3,           # Depth of GAT
                             "reparam_weight": False,   # Reparametrise outputs
                             "reparam_pos_enc": False,
                             "weight_bias": "zero",     # Zero: zero init. # Off, disable bias completely # Anything else: default init
                             "pos_enc_bias": "zero",
                             "load_d2v": True,          # Load pretrained datset2vec
                             "freeze_d2v": False,       # Continue training datset2vec
                             "pos_depth": 2,            # Depth of positional encoder.
                             "norm_lin": True,          # Normalise weights by dividing by L2 norm. final classification weight
                             "norm_weights": True,      # GAT weights
                             "model_load": "model_main"},   # Which D2V to load from

                 "Optim": {"lr": 3e-4},

                 "DL_params": {"bs": 3,
                               "num_rows": 10,
                               "num_targets": 5,
                               },

                 "Settings": {"num_epochs": 10000,      # Number of trainin epochs
                              "val_duration": 400,      # Number of batches of validation
                              "val_interval": 2000,     # Number of batches to train for each epoch
                              "dataset": "total",       #
                              },

                 "MLP_DL_params": {"noise_std": 0.3,
                                   "pre_sample_weights": False,
                                   "hidden_dim": 3,
                                   "num_causes": 2,
                                   "num_layers": 3,
                                   "is_causal": True,
                                   "dropout_prob": 0.5,
                                   "init_std": 1,
                                   "pre_sample_causes": False,
                                   "in_clique": True,
                                   "is_effect": True,
                                   "sort_features": True},
                 }

    save_dict["NN_dims"]["gat_in_dim"] = save_dict["NN_dims"]["pos_enc_dim"] + 1
    with open("./Fewshot/defaults.toml", "w") as f:
        toml.dump(save_dict, f)


if __name__ == "__main__":
    print("Resetting config to defaults")
    write_toml()
    get_config()
