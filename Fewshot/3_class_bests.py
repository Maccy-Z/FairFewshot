import pickle
import os
import pandas as pd
import re

def get_accuracy(model_name, ds_name, num_rows):
    if num_rows == 10 and not model_name == "STUNT":
        with open(f'./datasets/data/{ds_name}/3_class_only.dat', "r") as f:
            lines = f.read()
            lines = lines.split("\n")
        for config in lines:
            if config.startswith(f'{model_name}'):
                config = config.split(",")

                mean, std = float(config[-2]), float(config[-1])
                return mean, std
    else:
        with open(f'./datasets/data/{ds_name}/3_class_base.dat', "r") as f:
            lines = f.read()
            lines = lines.split("\n")
        for config in lines:
            if config.startswith(f'{model_name},{num_rows}'):
                config = config.split(",")

                mean, std = float(config[-2]), float(config[-1])
                return mean, std

    raise FileNotFoundError(f"Requested config does not exist: {model_name}, {ds_name}, {num_rows=}")


def get_matching_files(folder_path, num_rows):
    pattern = r"^.+_\d+\.pkl$"
    matching_files = []
    all_files = os.listdir(folder_path)

    # Define a regex pattern to match the desired format
    pattern = r"^.+_(\d+)\.pkl$"

    matching_files = []
    for f in all_files:
        match = re.match(pattern, f)
        if match and int(match.group(1)) == num_rows:
            matching_files.append(f)
    return matching_files


def main(num_rows):
    datasets = sorted([d for d in os.listdir("./datasets/data/") if os.path.isdir(os.path.join("./datasets/data/", d))])
    model_names = ["LR", "KNN", "R_Forest", "CatBoost", "FTTransformer", "STUNT", "SVC", "TabPFN"]


    all_results = {}
    for ds in datasets:
        ds_result = {}
        for model in model_names:
            try:
                m, std = get_accuracy(model, ds_name=ds, num_rows=num_rows)
                ds_result[model] = m

            except FileNotFoundError as e:
                print(e)

        all_results[ds] = ds_result
    df = pd.DataFrame.from_dict(all_results, orient="index")
    df.rename(columns={None: 'Dataset'}, inplace=True)

    flat_dir = "./Results/3.1_class/6"
    wanted_files = get_matching_files(flat_dir, num_rows=num_rows)

    print(wanted_files)

    flat_results = []
    for file in wanted_files:
        with open(f'{flat_dir}/{file}', "rb") as f:
            results = pickle.load(f)[-1]
            flat_results.append(results)

    trained_results = pd.concat(flat_results).reset_index(drop=True)

    flat = trained_results[trained_results["model"] == "FLAT"]
    flat = flat.pivot(index="data_name", columns="model", values="acc")

    adapt = trained_results[trained_results["model"] == "FLAT_maml"]
    adapt = adapt.pivot(index="data_name", columns="model", values="acc")
    df = pd.merge(df, adapt, left_index=True, right_index=True)
    # df = pd.merge(df, flat, left_index=True, right_index=True)
    print()
    print()
    print(flat.index)
    df = df.reset_index(drop=True)
    # df = df.idxmax(axis=1).value_counts()
    # print(df.sum())


if __name__ == "__main__":
    main(5)


