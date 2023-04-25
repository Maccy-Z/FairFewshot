import torch
import sys
import os
import torch_geometric

print("Pytorch version:", torch.__version__)
print("Torch Geometric version:", torch_geometric.__version__)

print("GPU available:", torch.cuda.is_available())
print("Python version:", sys.version)
print("Current working directory:", os.getcwd())