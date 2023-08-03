import pickle
import os



def get_accuracy(model_name, ds_name, num_rows, num_cols):
    with open(f'./datasets/data/{ds_name}/baselines.dat', "r") as f:
        lines = f.read()

    lines = lines.split("\n")

    for config in lines:
        if config.startswith(f'{model_name},{num_rows},{num_cols}'):
            config = config.split(",")

            mean, std = float(config[-2]), float(config[-1])
            return mean, std

    raise FileNotFoundError(f"Requested config does not exist: {model_name}, {ds_name}, {num_rows=}, {num_cols=}")


datasets = sorted([d for d in os.listdir("./datasets/data/") if os.path.isdir(os.path.join("./datasets/data/", d))])
model_names = ["LR", "KNN", "R_Forest", "CatBoost", "FTTransformer", "STUNT", "SVC", "TabNet"]


all_results = {}
for ds in datasets:
    ds_result = {}
    for model in model_names:
        try:
            m, std = get_accuracy(model, ds_name="adult", num_rows=10, num_cols=-3)
            ds_result[model] = (m, std)

        except FileNotFoundError as e:
            print(e)

    all_results[ds] = ds_result


print(all_results)
