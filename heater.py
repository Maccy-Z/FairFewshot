import torch

x = torch.rand([512, 512], device="cuda")
while True:
    x + 1

