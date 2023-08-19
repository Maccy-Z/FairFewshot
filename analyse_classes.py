#%%
import os
import pandas as pd

DATADIR = './datasets/data'

ds_names = os.listdir(DATADIR)
ds_names.remove('info.json')
if '.DS_Store' in ds_names:
    ds_names.remove('.DS_Store')

# ds_names = [
#     'acute-inflammation', 'acute-nephritis', 'arrhythmia',
#     'blood', 'breast-cancer', 'breast-cancer-wisc', 'breast-cancer-wisc-diag', 
#     'breast-cancer-wisc-prog', 'breast-tissue', 'cardiotocography-3clases', 
#     'dermatology', 'echocardiogram', 'heart-cleveland', 
#     'heart-hungarian', 'heart-switzerland', 'heart-va', 'hepatitis', 'horse-colic',
#     'ilpd-indian-liver', 'lymphography', 'mammographic', 
#     'parkinsons', 'post-operative', 'primary-tumor', 'spect', 'spectf', 
#     'statlog-heart', 'thyroid', 'vertebral-column-2clases'
# ]
# %%
ds_classes = {}
for ds_name in ds_names:
    labels = pd.read_csv(f'{DATADIR}/{ds_name}/labels_py.dat')
    n_classes = labels.iloc[:, 0].nunique()
    ds_classes[ds_name] = n_classes   
# %%
classes_df = pd.DataFrame({
    'ds_name': ds_classes.keys(),
    'n_classes': ds_classes.values()
})

# %%
def get_class_type(x):
    if x <= 3:
        return str(x)
    else:
        return '5+'
    
classes_df['n_class_type'] = classes_df['n_classes'].apply(lambda x: get_class_type(x))
classes_df['n_class_type'].value_counts()
# %%

# %%

# medical 38%
# all 55%
#%%
classes_df.n_classes.value_counts()
# %%

split = {}
split['0'] = {}
split['0']['test'] = list(classes_df[classes_df.n_classes == 2]['ds_name'].values)
split['0']['train'] = list(classes_df[classes_df.n_classes > 2]['ds_name'].values)
# %%
import toml

with open("./datasets/grouped_datasets/orig_binary", "w") as fp:
    toml.dump(split, fp) 
# %%
