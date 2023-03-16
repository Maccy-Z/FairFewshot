import torch

x = torch.tensor([False, False, False,  True, False, True, False])

y = torch.nonzero(x, as_tuple=True)[0]
print(y)
print(x)


