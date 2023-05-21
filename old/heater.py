import torch

x = torch.zeros([2048, 2048], device="cuda")

while True:
    x * x


