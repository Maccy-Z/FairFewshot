import numpy as np
from matplotlib import pyplot as plt


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


accs = proc_data(["70.65	70.995	71.02	69.665	71.0375	70.365	69.9125	68.42	68.87",
                  "70.41	70.645	69.7375	68.775	69.7475	69.5475	69.475	68.42	68.68",
                  "70.01	70.3125	67.155	67.5725	66.115	67.495	67.72	64.25	68.53",
                  "68.36	69.43	62.32	65.1225	60.46	63.175	64.495	59.11	67.87",
                  "66.84	67.32	55.12	60.6475	53.625	56.135	58.195	53.92	65.54"
                  ])

errors = proc_data(["0.144	0.145	0.278	0.263	0.261	0.263	0.263	0.13	0.30",
                    "0.128	0.120	0.260	0.263	0.263	0.263	0.263	0.13	0.21",
                    "0.129	0.131	0.260	0.268	0.273	0.273	0.266	0.14	0.21",
                    "0.125	0.121	0.290	0.276	0.283	0.280	0.273	0.14	0.21",
                    "0.139	0.151	0.248	0.280	0.293	0.293	0.285	0.15	0.21"
                    ])

num_1s = [5, 6, 7, 8, 9]
model_names = ["FLAT", "FLATadapt", "LR", "KNN", "RForest", "CatBoost", "FTT", "TabPFN", "Iwata"]
for model_num, (model_accs, model_errs) in enumerate(zip(accs, errors)):
    plt.errorbar(num_1s, model_accs, yerr=model_errs, label=model_names[model_num])

plt.legend()
plt.xlim(min(num_1s), max(num_1s))
plt.xticks(range(min(num_1s), max(num_1s) + 1))
plt.ylabel("Accuracy / %")
plt.xlabel("k, No. of most common class")

plt.plot([5, 10], [70.3, 70.3], c="C0", linestyle="--", alpha=0.7)
plt.plot([5, 10], [70.6, 70.6], c="C1", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("./test.pdf")

plt.show()
