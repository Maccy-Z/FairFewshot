import torch

x = torch.randn(3, 4)

print(x)

torch.nn.init.kaiming_uniform_(x)
print(x)

