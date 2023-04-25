#%%
import random
import numpy as np
import toml
from Fewshot.AllDataloader import MyDataSet

random.seed(123)
np.random.seed(123)

medical_datasets = [
    'acute-inflammation', 'acute-nephritis', 'arrhythmia',
    'blood', 'breast-cancer', 'breast-cancer-wisc', 'breast-cancer-wisc-diag', 
    'breast-cancer-wisc-prog', 'breast-tissue', 'cardiotocography-3clases', 
    'dermatology', 'echocardiogram', 'fertility', 'heart-cleveland', 
    'heart-hungarian', 'heart-switzerland', 'heart-va', 'hepatitis', 'horse-colic',
    'ilpd-indian-liver', 'lung-cancer', 'lymphography', 'mammographic', 
    'parkinsons', 'post-operative', 'primary-tumor', 'spect', 'spectf', 
    'statlog-heart', 'thyroid', 'vertebral-column-2clases'
]

total = len(medical_datasets)
n_test = int(total * 0.2)

forbidden_test_datasets = [
    'heart-cleveland', 'heart-hungarian', 'heart-switzerland', 'heart-va',
    'breast-cancer-wisc-diag', 'breast-cancer-wisc-prog', 'arrhythmia'
]

valid_test_datasets = list(set(medical_datasets).difference(set(forbidden_test_datasets)))
idx = np.random.permutation(len(valid_test_datasets))
valid_test_datasets = np.array(valid_test_datasets)[idx]

test_splits = [valid_test_datasets[k*4:(k+1)*4] for k in range(6)]


splits = {}

for i in range(6):
    train_split = list(set(medical_datasets).difference(set(test_splits[i])))
    test_split = test_splits[i]
    datasets = [
        MyDataSet(d, num_rows=5, num_targets=5, binarise=True, split="all") 
        for d in test_split]
    max_test_col = max([d.ds_cols - 1 for d in datasets])
    i = str(i)
    splits[i] = {}
    splits[i]['train'] = train_split
    splits[i]['test'] = list(set(medical_datasets).difference(set(train_split)))
    splits[i]['max_col'] = max_test_col
 
with open("./datasets/grouped_datasets/med_splits", "w") as fp:
    toml.dump(splits, fp) 


# %%
