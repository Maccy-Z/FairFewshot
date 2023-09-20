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



    raise FileNotFoundError(f"Requested config does not exist: {model_name}, {ds_name}, {num_rows=}, {num_cols=}")


datasets = sorted([d for d in os.listdir("./datasets/data/") if os.path.isdir(os.path.join("./datasets/data/", d))])
model_names = ["LR", "KNN", "R_Forest", "CatBoost", "FTTransformer", "STUNT", "SVC", "TabNet", "TabPFN", "Iwata"]

all_results = {}
for ds in datasets:
    ds_result = {}
    for model in model_names:
        try:
            m, std = get_accuracy(model, ds_name=ds, num_rows=3, num_cols=-3)
            ds_result[model] = m

        except FileNotFoundError as e:
            print(e)

    all_results[ds] = ds_result
df = pd.DataFrame.from_dict(all_results, orient="index")

get = "FLAT"

with open("./Results/old_final/8.1/raw.pkl", "rb") as f:
    results = pickle.load(f)

    results = results[results["model"].isin(["FLAT", "FLAT_maml"])]
    s1 = results.drop("num_cols", axis=1)

    # print(s1)
    # s1 = r.drop("model", axis=1)

with open("./Results/old_final/22/raw.pkl", "rb") as f:
    results = pickle.load(f)
    results = results[results["model"].isin(["FLAT", "FLAT_maml"])]
    s2 = results.drop("num_cols", axis=1)
    # s2 = r.drop("model", axis=1)

with open("./Results/old_final/23/raw.pkl", "rb") as f:
    results = pickle.load(f)
    results = results[results["model"].isin(["FLAT", "FLAT_maml"])]
    s3 = results.drop("num_cols", axis=1)
    # s3 = r.drop("model", axis=1)

with open("./Results/old_final/24/raw.pkl", "rb") as f:
    results = pickle.load(f)
    results = results[results["model"].isin(["FLAT", "FLAT_maml"])]
    s0 = results.drop("num_cols", axis=1)
    # s0 = r.drop("model", axis=1)

s_all = pd.concat([s0, s1, s2, s3]).reset_index(drop=True)

for ds, results in all_results.items():
    s = s_all[(s_all["data_name"] == ds)].squeeze()

    if s.any().all():
        flat_stats = s[s["model"] == "FLAT"]
        maml_stats = s[s["model"] == "FLAT_maml"]

        results["FLAT_diff"] = flat_stats["acc"].iloc[0] - max(list(results.values()))
        results["FLAT_maml"] = maml_stats["acc"].iloc[0]
        results["FLAT"] = flat_stats["acc"].iloc[0]
        # results["diff_diff"] = results["FLAT_maml"] - results["FLAT"]

df = pd.DataFrame.from_dict(all_results, orient="index")

df = df.sort_values(by='FLAT_diff', ascending=True)

cols = df.columns.tolist()
b, c = cols.index('FLAT_diff'), cols.index(get)
cols[b], cols[c] = cols[c], cols[b]
df = df[cols]
df = df.round(3)

print()
df = df.sort_index()
df.rename(columns={"FLAT_maml": "FLATadapt"}, inplace=True)
df.drop(columns="FLAT_diff", inplace=True)

# Replace certain rows with NaN
rows = ["hill-valley", "musk-1", "low-res-spect", "musk-2", "arrhythmia", "semeion"]
for row in rows:
    df.loc[row, "TabPFN"] = "NaN"

print()
print(df.to_string())

df.to_csv("./all_binary_results3")

# print(df["LR"].mean())
#
# print("LR", df["LR"].mean())
# print("R_Forest", df['R_Forest'].mean())
# print("SVC", df['SVC'].mean())
# print("CatBoost", df['CatBoost'].mean())
# print("TabPFN", df['TabPFN'].mean())
# print("Iwata", df['Iwata'].mean())
#
# print("FLAT", df['FLAT'].mean())
# print("FLATadapt", df['FLATadapt'].mean())



# print(df.index.tolist())
# print(df['diff_diff'].tolist())
# print(df["FLAT_diff"].tolist())


# from scipy.stats import ttest_ind
#
# DATADIR = './datasets/data'
#
# data_dim_df = pd.DataFrame(index=df.index)
# data_dim_df['n_classes'] = None
#
# print(data_dim_df)
#
# # Get the number of columns and no. of original classes of the dataset
# for ds_name in data_dim_df.index:
#     labels = pd.read_csv(f'{DATADIR}/{ds_name}/labels_py.dat', header=None)
#     data = pd.read_csv(f'{DATADIR}/{ds_name}/{ds_name}_py.dat', header=None)
#     n_classes = labels.iloc[:, 0].nunique()
#     n_cols = data.shape[1]
#     #data_dim_df.loc[ds_name, 'n_cols'] = n_cols
#     data_dim_df.loc[ds_name, 'n_classes'] = n_classes
#
# for baseline in df.columns:
#     compare_df = df[['FLAT', baseline]]
#     compare_df["diff"] = df['FLAT'] - df[baseline]
#
#     compare_df = compare_df.join(data_dim_df)
#
#     # Test the hypothesis that the number of original classes influences the performance
#     a = compare_df[compare_df.n_classes == 2]['diff']
#     b = compare_df[compare_df.n_classes > 2]['diff']
#
#     print("FInal result")
#     print(baseline, ttest_ind(a, b).pvalue)
#     print(a.mean(), b.mean())
#
