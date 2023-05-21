import torch

x = torch.zeros([1024, 1024], device="cuda")

while True:
    torch.matmul(x, x)


