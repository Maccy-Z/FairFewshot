import os

base_dir = "./data"
datasets = sorted([f for f in os.listdir(base_dir) if os.path.isdir(f'{base_dir}/{f}')])


print(datasets)
print(len(datasets))