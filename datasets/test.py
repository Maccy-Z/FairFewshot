import os

base_dir = "./data"
datasets = sorted([f for f in os.listdir(base_dir) if os.path.isdir(f'{base_dir}/{f}')])

med_ds = ['acute-inflammation', 'acute-nephritis', 'arrhythmia',
            'blood', 'breast-cancer', 'breast-cancer-wisc', 'breast-cancer-wisc-diag',
            'breast-cancer-wisc-prog', 'breast-tissue', 'cardiotocography-3clases',
            'dermatology', 'echocardiogram', 'fertility', 'heart-cleveland',
            'heart-hungarian', 'heart-switzerland', 'heart-va', 'hepatitis', 'horse-colic',
            'ilpd-indian-liver', 'lung-cancer', 'lymphography', 'mammographic',
            'parkinsons', 'post-operative', 'primary-tumor', 'spect', 'spectf',
            'statlog-heart', 'thyroid', 'vertebral-column-2clases']

nom_med_ds = [ds for ds in datasets if ds not in med_ds]


print(nom_med_ds)
print(len(nom_med_ds))

print(datasets)
print(len(datasets))