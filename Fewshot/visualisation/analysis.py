import torch
import numpy as np
import pandas as pd
import os
import random
import toml
from itertools import islice
import matplotlib.pyplot as plt

from ..AllDataloader import SplitDataloader
from scipy.stats import linregress, pearsonr

# datasets = os.listdir("/mnt/storage_ssd/fewshot_learning/FairFewshot/datasets/data")
# datasets = sorted([f for f in datasets if os.path.isdir(f'/mnt/storage_ssd/fewshot_learning/FairFewshot/datasets/data/{f}')])
# print(datasets)
#
# sizes = []
# count = 0
# for ds in []:
#     try:
#         dl = SplitDataloader(
#             bs=1, num_rows=9, balance=1, num_targets=5, num_cols=-3, ds_group=ds, ds_split="train",
#         )
#         sizes.append(len(dl))
#         count += 1
#
#     except IndexError as e:
#         print(e)
#         pass
#
# # Acc vs baseline
# accs = [-0.209, -0.127, -0.12, -0.117, -0.092, -0.091, -0.089, -0.082, -0.081, -0.078, -0.076, -0.07, -0.07, -0.069, -0.064, -0.053, -0.052, -0.051, -0.05,
#         -0.05, -0.043, -0.041, -0.039, -0.038, -0.036, -0.035, -0.034, -0.033, -0.033, -0.031, -0.027, -0.027, -0.026, -0.026, -0.026, -0.025, -0.025, -0.025,
#         -0.022, -0.02, -0.02, -0.017, -0.017, -0.017, -0.016, -0.015, -0.014, -0.014, -0.013, -0.011, -0.01, -0.01, -0.009, -0.008, -0.008, -0.008, -0.007,
#         -0.007, -0.007, -0.006, -0.006, -0.005, -0.004, -0.004, -0.004, -0.003, -0.003, -0.003, -0.002, -0.002, -0.002, -0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.001,
#         0.001, 0.002, 0.002, 0.003, 0.003, 0.004, 0.004, 0.005, 0.005, 0.006, 0.006, 0.007, 0.009, 0.009, 0.01, 0.011, 0.012, 0.013, 0.013, 0.014, 0.016, 0.017,
#         0.017, 0.018, 0.019, 0.02, 0.02, 0.021, 0.021, 0.022, 0.03, 0.031, 0.033, 0.047, 0.047, 0.052, 0.077]
#
# plt.scatter(sizes, accs, marker="x")
# plt.xlabel("N cols")
# plt.ylabel("Acc diff vs base")
# plt.xlim([0, 100])
# plt.title("Accuracy diff vs baseline")
# plt.show()


diff_diffs = [-0.032, -0.031, -0.026, -0.022, -0.019, -0.019, -0.016, -0.016, -0.016, -0.015, -0.014, -0.013, -0.013, -0.011, -0.01, -0.01, -0.01, -0.009, -0.008, -0.008, -0.008, -0.008, -0.007, -0.007, -0.006, -0.006, -0.006, -0.006, -0.006, -0.005, -0.005, -0.005, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.003, -0.003, -0.003, -0.003, -0.003, -0.003, -0.003, -0.003, -0.002, -0.002, -0.002, -0.001, -0.001, -0.001, 0.0, 0.001, 0.001, 0.001, 0.001, 0.003, 0.003, 0.004, 0.004, 0.004, 0.005, 0.005, 0.005, 0.005, 0.006, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.008, 0.008, 0.008, 0.009, 0.009, 0.009, 0.009, 0.009, 0.01, 0.011, 0.011, 0.011, 0.012, 0.012, 0.013, 0.013, 0.014, 0.014, 0.014, 0.015, 0.015, 0.016, 0.017, 0.017, 0.017, 0.02, 0.02, 0.021, 0.023, 0.024, 0.024, 0.026, 0.031, 0.032, 0.032, 0.037, 0.043, 0.051, 0.059, 0.062, 0.065, 0.082]
flat_diffs = [0.047, 0.013, -0.004, 0.003, 0.018, -0.007, -0.013, -0.026, -0.003, -0.02, -0.01, -0.209, 0.017, -0.002, -0.017, 0.011, 0.021, 0.009, 0.02, -0.02, 0.014, 0.013, 0.004, -0.035, 0.021, 0.012, -0.003, -0.011, -0.017, 0.0, 0.005, -0.092, -0.017, -0.007, -0.008, -0.014, -0.027, -0.025, 0.033, 0.047, 0.003, -0.002, -0.008, 0.0, -0.034, -0.008, -0.006, 0.02, -0.009, 0.006, -0.007, 0.001, -0.025, -0.004, 0.002, -0.016, -0.026, 0.016, 0.004, 0.005, -0.07, -0.003, -0.043, -0.036, 0.03, -0.004, -0.014, -0.006, 0.019, 0.031, 0.001, -0.01, -0.027, -0.127, -0.078, 0.0, -0.05, 0.022, -0.002, -0.076, 0.001, -0.053, -0.05, -0.026, -0.07, 0.009, 0.007, -0.005, -0.117, -0.015, -0.12, -0.069, 0.01, -0.0, 0.006, -0.039, 0.052, -0.022, 0.0, -0.052, -0.041, -0.051, 0.017, 0.002, -0.033, -0.033, -0.031, -0.064, -0.089, -0.025, -0.081, 0.077, -0.091, -0.038, -0.082]



plt.scatter(flat_diffs, diff_diffs)
plt.ylabel("FLATadapt vs FLAT")
plt.xlabel('Flat vs baseline')

diff_diff, flat_diff = [], []
for d, f in zip(diff_diffs, flat_diffs):
    if -0.2 < f < 0.075 and d < 0.08:
        diff_diff.append(d)
        flat_diff.append(f)

result = linregress(flat_diff, diff_diff)
correlation_coefficient, p_value = pearsonr(flat_diff, diff_diff)

regression_ys = [result.slope * x + result.intercept for x in flat_diffs]

plt.plot(flat_diffs, regression_ys, color='red', label='Regression Line')


plt.show()



print(f"Correlation coefficient: {correlation_coefficient}")
print(f"Slope (m): {result.slope}")
print(f"Intercept (c): {result.intercept}")
r_squared = result.rvalue ** 2
print(f"R^2 value: {r_squared}")


