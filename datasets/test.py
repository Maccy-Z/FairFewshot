import os

base_dir = "./data"
datasets = sorted([f for f in os.listdir(base_dir) if os.path.isdir(f'{base_dir}/{f}')])

for dataset in datasets:
    new_baselines = []
    with open(f'./data/{dataset}/baselines.dat', "r") as f:
        old_baselines = f.readlines()

    with open(f'./data/{dataset}/base_RF_fix.dat', "r") as f:
        RF_baseline = f.readlines()

    for old_base in old_baselines:
        if not old_base.startswith("R_Forest"):
            new_baselines.append(old_base)

    new_baselines += RF_baseline
    new_baselines = sorted(new_baselines)
    #nprint(new_baselines)

    # exit(2)


    new_baselines = sorted(new_baselines)

    with open(f'./data/{dataset}/baselines.dat', "w") as f:
        for b in new_baselines:
            f.write(b)





