import toml


def get_config():
    with open("./Fewshot/defaults.toml", "r") as f:
        config = toml.load(f)

    return config


def write_toml():
    save_dict = {"NN_dims": {"pos_enc_dim": 15,
                             "set_h_dim": 64,
                             "set_out_dim": 32,
                             "weight_hid_dim": 64,
                             "gat_heads": 2,
                             "gat_hid_dim": 128,
                             "gat_out_dim": 16,
                             "d2v_layers": [3, 2, 3],
                             "gen_layers": 2,
                             "gat_layers": 2,
                             "reparam_weight": False,
                             "reparam_pos_enc": False,
                             "weight_bias": "zero",
                             "pos_enc_bias": "zero",
                             "load_d2v": False,
                             "freeze_d2v": False,
                             "pos_depth": 2,
                             "norm_lin": False,
                             "norm_weights": False,
                             "model_load": "model_main"},

                 "Optim": {"lr": 3e-4},

                 "DL_params": {"bs": 5,
                               "num_rows": 10,
                               "num_targets": 10,
                               "flip": False,
                               "num_cols": 5,
                               "data_names": ['mammographic', 'teaching'],
                               "shuffle_cols": True,
                 },

                 "Settings": {"num_epochs": 100,
                              "val_duration": 100,
                              "val_interval": 1000,
                              "dataset": "dummy",
                              },

                 "MLP_DL_params": {"noise_std": 0.3,
                                   "pre_sample_weights": False,
                                   # "activation": "Sigmoid",
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
