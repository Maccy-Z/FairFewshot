from Fewshot.dataloader import AdultDataLoader
import torch

dl = AdultDataLoader(bs=2, num_rows=10, num_target=3, flip=False)


males = torch.sum(dl.data[:, 9] > 0)
fem = torch.sum(dl.data[:, 9] < 0)


print(males - fem)

