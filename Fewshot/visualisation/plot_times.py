import numpy as np
from matplotlib import pyplot as plt

plt.figure(figsize=(4,4))

accs = """0.292	3.5	0.229	0.133	0.0885	25.2	11.4	97.5	30.7	10.1	0.256	17.5
0.304	3.97	0.245	0.115	0.0894	25.2	12.1	120.8	31.8	9.92	0.289	25.7
0.325	4.81	0.267	0.115	0.107	25.3	12.5	111.1	32.9	10	0.281	27.5
0.342	6.51	0.283	0.12	0.109	24.9	13.1	107.9	34.3	11	0.268	24.9
0.359	8.72	0.303	0.101	0.0888	25.2	13.5	108.1	35.2	11.8	0.267	29.6
0.611	23.1	0.364	0.137	0.0918	25.3	16.8	118.2	40.5	13.1	0.342	36.6
1.09	82	0.423	0.0976	0.091	25.3	23	127.6	52.1	14.1	0.364	59.9
6.4	272	0.497	0.0998	0.0878	25.1	67.5	126.8	86.6	16.8	0.438	400
26.3	400	0.566	0.108	0.09	25.2	114	134.8	263	24.9	0.528	400"""

rows = accs.strip().split("\n")
table = [row.split("\t") for row in rows]
table = np.array(table, dtype=float).T

ds_names = "FLAT	FLATadapt	LR	KNN	SVC	R_Forest	CatBoost	TabNet	FTT	STUNT	Itawa	TabPFN"
ds_names = ds_names.split()
print(ds_names)
print(table.shape, len(ds_names))

x_label = [5, 10, 15, 20, 25, 50, 100, 200, 400]
for ds, t in zip(ds_names, table, strict=True):

    t = [a if a < 200 else 200 for a in t]
    if ds in ["FLAT", "FLATadapt"]:
        plt.plot(t, label=ds)
    else:
        plt.plot(t, label=ds, alpha=0.5, linestyle="--")


plt.legend(loc='upper left')
plt.xticks(np.arange(9), x_label)
plt.ylim(0, 150)

plt.ylabel("t / s")
plt.xlabel("Ncol")
plt.tight_layout()
plt.show()



