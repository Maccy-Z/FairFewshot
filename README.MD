# FLAT
## Tabular Few-Shot Generalization Across Heterogeneous Feature Spaces

### How to run the main experiments:

1. Download the data from this [google drive](https://drive.google.com/file/d/11FAi02DqcWpqHUwKnc8JWSQHIfpwy6R9/view?usp=sharingand) and put the contents in the `datasetes/data` directory

2. Set the configuration file in `Fewshot/configs.toml`

    Most of the settings here are pre-selected model hyperparamters. 
    To train the model on the medical datasets set (Section 4.1):
    ```
    fold_no={fold_no}
    folds_file="med_folds"
    ```

    To train the model on all UCI datasets set (Section 4.2):
    ```
    fold_no={fold_no}
    folds_file="folds"
    ```

    `{fold_no}` should be an integer between 0 to $N-1$, which indicates which of the $N$-folds within the dataset collection is held-out for testing.
    For the medical example, $N=10$. For all datasets $N=4$.

3. To train the FLAT model, run `python Fewshot/main.py`. Models are automatically saved in `saves/{run_no}`.

4. To evaluate FLAT models and the baselines run `python Fewshot/comparison.py`. 

    Select which FLAT model from saved models to evaluate by setting `load_no=[{run_no}]` and `num_rows={num_rows}`, where `num_rows` is the number of meta rows used during inference, $N^{meta}$. 

    `Fewshot/comparison.py` contains the code for the baselines, including the hyperparamters used in all our testing. Baselines can also be evaluated by setting line 470, e.g. 
    ```
    models = [Iwata(0), BasicModel("LR")]
    ``` 

    For Iwata, we have 4+10 pretrained models, saved in `iwata/{model_no}`. The first 4 models (from 0 to 4) correspond to each of the 4 folds from Section 4.2. 10 models in `iwata/med_iwata/` correspond to each of the 10 folds from Section 4.1. To load these models, use `Iwata({model_no})`. (To load iwata models in the `med_iwata` subfolder, change the `save_dir` path in line 371 on `Fewshot/comparison.py`). The code to train Iwata models is in `iwata.py`. 

