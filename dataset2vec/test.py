import torch

x = torch.arange(4.).view(1, 2, 2)

print(x)

print(torch.nn.functional.normalize(x, dim=-1))

