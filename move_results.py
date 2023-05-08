import shutil
import os


saves = [f for f in sorted(os.listdir("./data")) if os.path.isdir(f'./data/{f}')][0]

with open(f'./datasets/data/{saves}/baselines.dat', "r") as f:
    lines = f.read()

    print(lines.split("\n"))





