FLAT 

Instructions for running the code:

First, configure which dataset to train on in defaults.toml. 
Most of the settings here are pre-selected model hyperparamters. Choose which dataset to use by setting ds_group (ln30), split_file(ln32) and dataset (ln38). 
Setting dataset="total" (Section 4.2) uses all UCI datasets, or dataset="med_splits2" uses medical datasets only (Section 4.1). The number of training meta rows is set using num_rows (ln28)

To train the FLAT model, run main.py. Models are automatically saved in /saves/{run_no}. 

Models can be evaluated by running comparison.py. Select which model to evaluate by setting load_no=[{run_no}] and num_rows={num_rows}.
Additionally, baselines can also be evaluated by setting line 488, e.g. models = [Iwata(0), BasicModel("LR")]. 

comparison.py contains the code for the baselines, including the hyperparamters used in all our testing. 
For Iwata, we have 4 pretrained models, saved in ./iwata/{model_no}, corresponding to each of the 4 dataset folds in Section 4.2. To load these models, use Iwata({model_no}). 

The code to train Iwata models is in iwata.py. 

