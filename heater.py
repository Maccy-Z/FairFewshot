import torch

x = torch.rand([4096, 4096], device="cuda")
while True:
    x + 1

