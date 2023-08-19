#%%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('ticks')
sns.set_palette('Set2')

def proc_data(xs):
    xs = [x.split() for x in xs]

    outs = []
    for n_shot_row in xs:
        row = []
        for acc in n_shot_row:
            row.append(float(acc))
        outs.append(row)

    outs = np.array(outs).T

    return outs


accs = proc_data(["70.4125	70.995	71.02	69.665	70.2975	69.1175	69.9125",
                    "70.0125	70.645	69.7375	68.775	68.98	68.625	69.475",
                    "68.3625	70.3125	67.155	67.5725	65.905	66.9075	67.72",
                    "68.125	69.43	62.32	65.1225	60.6575	63.175	64.495",
                    "66.84	67.32	55.12	60.6475	53.625	56.135	58.195"])

errors = proc_data(["0.144	0.145	0.278	0.263	0.261	0.263	0.263",
                    "0.128	0.120	0.260	0.263	0.263	0.263	0.263",
                    "0.129	0.131	0.260	0.268	0.273	0.273	0.266",
                    "0.125	0.121	0.290	0.276	0.283	0.280	0.273",
                    "0.139	0.151	0.248	0.280	0.293	0.293	0.285",])

colors = ['#a2d1f2', '#50a7e6', 'C1', 'C6', 'C3', 'C4', 'C5']
num_1s = [5, 6, 7, 8, 9]
model_names = ["FLAT", "FLAT-MAML", "LR", "KNN", "RandForest", "CatBoost", "FT-Transformer"]
fig, ax = plt.subplots(figsize=(6, 4))
for model_num, (model_accs, model_errs) in enumerate(zip(accs, errors)):
    ax.errorbar(
        num_1s, model_accs, yerr=model_errs, 
        label=model_names[model_num],
        color = colors[model_num]
    )

ax.legend()
ax.set_xlim(min(num_1s), max(num_1s))
ax.set_xticks(range(min(num_1s), max(num_1s)+1))
ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("$k$ - no. of the majority class")

ax.plot([5, 10], [70.3, 70.3], c='#a2d1f2', linestyle="--", alpha=0.7)
ax.plot([5, 10], [70.6, 70.6], c='#50a7e6', linestyle="--", alpha=0.7)
plt.savefig('figures/imbalanced_class')
plt.show()


# %%
