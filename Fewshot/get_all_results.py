import pickle
import os
import pandas as pd

def get_accuracy(model_name, ds_name, num_rows, num_cols):
    with open(f'./datasets/data/{ds_name}/baselines.dat', "r") as f:
        lines = f.read()
        lines = lines.split("\n")

    for config in lines:
        if config.startswith(f'{model_name},{num_rows},{num_cols}'):
            config = config.split(",")

            mean, std = float(config[-2]), float(config[-1])
            return mean, std

    with open(f'./datasets/data/{ds_name}/base_tabpfn.dat', "r") as f:
        lines = f.read()
        lines = lines.split("\n")

    for config in lines:
        if config.startswith(f'{model_name}'):
            config = config.split(",")

            mean, std = float(config[-2]), float(config[-1])
            return mean, std

    raise FileNotFoundError(f"Requested config does not exist: {model_name}, {ds_name}, {num_rows=}, {num_cols=}")


datasets = sorted([d for d in os.listdir("./datasets/data/") if os.path.isdir(os.path.join("./datasets/data/", d))])
model_names = ["LR", "KNN", "R_Forest", "CatBoost", "FTTransformer", "STUNT", "SVC", "TabNet", "TabPFN"]


all_results = {}
for ds in datasets:
    ds_result = {}
    for model in model_names:
        try:
            m, std = get_accuracy(model, ds_name=ds, num_rows=10, num_cols=-3)
            ds_result[model] = m

        except FileNotFoundError as e:
            print(e)

    all_results[ds] = ds_result
df = pd.DataFrame.from_dict(all_results, orient="index")

get = "FLAT"

with open("./Results/old_final/2/raw.pkl", "rb") as f:
    results = pickle.load(f)

    results = results[results["model"].isin(["FLAT", "FLAT_maml"])]
    s1 = results.drop("num_cols", axis=1)

    #print(s1)
    #s1 = r.drop("model", axis=1)

with open("./Results/old_final/10/raw.pkl", "rb") as f:
    results = pickle.load(f)
    results = results[results["model"].isin(["FLAT", "FLAT_maml"])]
    s2 = results.drop("num_cols", axis=1)
    #s2 = r.drop("model", axis=1)


with open("./Results/old_final/11/raw.pkl", "rb") as f:
    results = pickle.load(f)
    results = results[results["model"].isin(["FLAT", "FLAT_maml"])]
    s3 = results.drop("num_cols", axis=1)
    #s3 = r.drop("model", axis=1)


with open("./Results/old_final/4.1/raw.pkl", "rb") as f:
    results = pickle.load(f)
    results = results[results["model"].isin(["FLAT", "FLAT_maml"])]
    s0 = results.drop("num_cols", axis=1)
    #s0 = r.drop("model", axis=1)



s_all = pd.concat([s0, s1, s2, s3]).reset_index(drop=True)


for ds, results in all_results.items():
    s = s_all[(s_all["data_name"] == ds)].squeeze()

    if s.any().all():
        flat_stats = s[s["model"] == "FLAT"]
        maml_stats = s[s["model"] == "FLAT_maml"]

        results["FLAT_diff"] = flat_stats["acc"].iloc[0] - max(list(results.values()))
        results["FLAT_maml"] = maml_stats["acc"].iloc[0]
        results["FLAT"] = flat_stats["acc"].iloc[0]
        results["diff_diff"] = results["FLAT"] - results["FLAT_maml"]


df = pd.DataFrame.from_dict(all_results, orient="index")


df = df.sort_values(by='diff_diff', ascending=True)

cols = df.columns.tolist()
b, c = cols.index('FLAT_diff'), cols.index(get)
cols[b], cols[c] = cols[c], cols[b]
df = df[cols]
df = df.round(3)


print(df.index.tolist())
print(df['diff_diff'].tolist())

# df = df.to_string()
# print(df)

