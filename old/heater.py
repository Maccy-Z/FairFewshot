import torch

x = torch.zeros([1024, 1024], device="cuda")

while True:
    x + 1

