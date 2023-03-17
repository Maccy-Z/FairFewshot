import torch
x = torch.tensor([1, 2, 3])


y = x.tile((2,))

print(y)
