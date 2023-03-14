import torch
import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, data_name):
        self.data_name = data_name
        self.data, self.labels = self.get_data()

    def get_data(self):
        """
        Dataset format: {folder}_py.dat             predictors
                        labels_py.dat               labels for predictors
                        folds_py.dat                test fold
                        validation_folds_py.dat     validation fold

        """
        datadir = f'./datasets/{self.data_name}'

        # get train fold
        folds = pd.read_csv(f"{datadir}/folds_py.dat", header=None)[0]
        folds = np.asarray(folds)
        # get validation fold
        vldfold = pd.read_csv(f"{datadir}/validation_folds_py.dat", header=None)[0]
        vldfold = np.asarray(vldfold)

        # read predictors
        predictors = pd.read_csv(f"{datadir}/{self.data_name}_py.dat", header=None)
        predictors = np.asarray(predictors)

        # read internal target
        targets = pd.read_csv(f"{datadir}/labels_py.dat", header=None)
        targets = np.asarray(targets)

        # get data folds
        data = {}
        data.update({'train': predictors[(1 - folds) == 1 & (vldfold == 0)]})
        data.update({'test': predictors[folds == 1]})
        data.update({'valid': predictors[vldfold == 1]})

        # get label folds
        labels = {}
        labels.update({'train': targets[(1 - folds) == 1 & (vldfold == 0)]})
        labels.update({'test': targets[folds == 1]})
        labels.update({'valid': targets[vldfold == 1]})

        print(predictors.shape, targets.shape)
        return data, labels


if __name__ == "__main__":
    ds = Dataset("adult")
