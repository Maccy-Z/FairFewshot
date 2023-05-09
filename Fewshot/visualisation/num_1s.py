import numpy as np

xs = ["70.995	71.02	69.665	70.2975	69.1175	69.9125",
"70.645	69.7375	68.775	68.98	68.625	69.475",
"70.67333333	67.155	67.5725	65.905	66.9075	67.72",
"69.97	62.32	65.1225	60.6575	63.175	64.495",
"50	55.12	60.6475	53.625	56.135	58.195"]

xs = [x.split() for x in xs]

accs = []
for n_shot_row in xs:
    row = []
    for acc in n_shot_row:
        row.append(float(acc))
    accs.append(row)

accs = np.array(accs).T
print(accs)

