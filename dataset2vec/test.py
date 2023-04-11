import torch
import sys
sys.path.append("/mnt/storage_ssd/FairFewshot/Fewshot")
from Fewshot.main import SetSetModel

load_state_dict = torch.load("/mnt/storage_ssd/FairFewshot/dataset2vec/model")
model = SetSetModel(64, 64, 3, [3, 2, 3, 2], reparam_weight=False, reparam_pos_enc=False, pos_enc_bias=False)
model.load_state_dict(load_state_dict, strict=False)

for fs in model.fs.parameters():
    fs.requires_grad = False
for gs in model.gs.parameters():
    gs.requires_grad = False
for hs in model.hs.parameters():
    hs.requires_grad = False

test_data = torch.rand([5, 10, 2])

test_out = model([test_data])
print(test_out)
