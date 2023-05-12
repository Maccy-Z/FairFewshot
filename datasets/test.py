import os

base_dir = "./data"
datasets = sorted([f for f in os.listdir(base_dir) if os.path.isdir(f'{base_dir}/{f}')])

for dataset in datasets:
    with open(f'./data/{dataset}/baselines.dat', "r") as f:
        new_baselines = f.readlines()

    with open(f'./data2/{dataset}/baselines.dat', "r") as f:
        stunt_baseline = f.readlines()

    for stunt_base in stunt_baseline:
        if stunt_base.startswith("STUNT"):
            new_baselines.append(stunt_base)


    new_baselines = sorted(new_baselines)

    with open(f'./data/{dataset}/baselines.dat', "w") as f:
        for b in new_baselines:
            f.write(b)

    os.remove(f'./data/{dataset}/baselines2.dat')




