import toml


def get_config():
    with open("./Fewshot/defaults.toml", "r") as f:
        config = toml.load(f)

    return config


def write_toml():
    save_dict = {"NN_dims": {"pos_enc_dim": 3,
                             "set_h_dim": 64,
                             "set_out_dim": 32,
                             "weight_hid_dim": 64,
                             "gat_heads": 2,
                             "gat_hid_dim": 128,
                             "gat_in_dim": 4,
                             "gat_out_dim": 16,
                             "d2v_layers": [3, 2, 3, 2],
                             "gen_layers": 2,
                             "gat_layers": 3,
                             "reparam_weight": True,
                             "reparam_pos_enc": True},

                 "Optim": {"lr": 3e-4},

                 "DL_params": {"bs": 3,
                               "num_rows": 10,
                               "num_targets": 5,
                               "flip": False},

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
                                   # "causes_mean": None,
                                   # "causes_std": None,
                                   "in_clique": True,
                                   "is_effect": True,
                                   "sort_features": True},

                 "Settings": {"num_epochs": 10000,
                              "print_interval": 100,
                              "save_dir": "",
                              "save_batch": 500,
                              }
                 }

    with open("./Fewshot/defaults.toml", "w") as f:
        toml.dump(save_dict, f)


if __name__ == "__main__":
    print("Resetting config to defaults")
    write_toml()
    get_config()
