import torch

x = torch.rand([1024, 1024], device="cuda")
while True:
    x + 1

