# Loads datasets.
import torch
import numpy as np
import pandas as pd


# Loads a single dataset. Split into train and valid.
class Dataset:
    def __init__(self, data_name, split="train", dtype=torch.float32, device="cpu"):
        self.data_name = data_name
        self.device = device
        self.dtype = dtype
        """      
        Data columns and num_catagories, 
        0, Age: 76
        1, Workclass: 9
        2, fnlwgt: 18299
        3, Education: 16
        4, Education-num: 16
        5, Marital: 7
        6, Occupation: 15
        7, Relation: 6 
        8, Race: 5
        9, Sex: 2
        10, Cap-gain: 118
        11, Cap-loss: 91
        12, Hr-per-wk: 92
        13, Native_cont: 41
        14, Income >50k: 2
        
        """
        datadir = f'./datasets/data_split/{self.data_name}'

        # Read data and labels
        full_dataset = pd.read_csv(f"{datadir}/{split}", header=None)
        full_dataset = np.asarray(full_dataset)

        # Combine validation and test folds together

        self.data = self._to_tensor(full_dataset)

    def _to_tensor(self, array: np.array):
        return torch.from_numpy(array).to(self.device).to(self.dtype)


# Randomly samples from dataset. Returns a batch of tables for use in the GNN
class AdultDataLoader:
    def __init__(self, *, bs, num_rows, num_target, flip=True, device="cpu", split="train"):
        self.bs = bs
        self.num_rows = num_rows
        self.num_target = num_target
        self.flip = flip

        self.ds = Dataset("adult", device=device, split=split)

        self.data = self.ds.data
        self.len = self.data.shape[0]
        self.cols = self.data.shape[1]

    # Pick out bs rows from dataset. Select 1 column to be target and random columns as predictors.
    # Only certain column can be targets since some are continuous.
    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        num_rows = self.num_rows + self.num_target

        while True:
            # Randomise data order
            permutation = torch.randperm(self.data.shape[0])
            data = self.data[permutation]

            allowed_targets = [0]  # [9, 14]
            cols = np.arange(self.cols)

            for st in torch.arange(0, self.len - num_rows * self.bs, num_rows * self.bs):
                # TODO: Allow flexible number of columns within a batch. Currently, each batch is fixed num_xs.
                self.num_xs = 10

                # Get target and predictor columns
                target_cols = np.random.choice(allowed_targets, size=self.bs)

                predict_cols = np.empty((self.bs, self.num_xs))
                for batch, target in enumerate(target_cols):
                    row_cols = np.setdiff1d(cols, target)
                    predict_cols[batch] = np.random.choice(row_cols, size=self.num_xs, replace=False)

                target_cols = torch.from_numpy(target_cols).int()
                predict_cols = torch.from_numpy(predict_cols).int()

                # Rows of data to select
                select_data = data[st:st + num_rows * self.bs]

                # Pick out wanted columns
                predict_idxs = predict_cols.repeat_interleave(num_rows, dim=0)
                target_idxs = target_cols.repeat_interleave(num_rows)

                xs = select_data[np.arange(self.bs * num_rows).reshape(-1, 1), predict_idxs]
                ys = select_data[np.arange(self.bs * num_rows), target_idxs]

                xs = xs.view(self.bs, num_rows, self.num_xs)  # [bs, num_rows, num_cols]
                ys = ys.view(self.bs, num_rows)  # [bs, num_rows]

                ys = self.binarise_data(ys)
                yield xs, ys

    # Convert target columns into a binary classification task.
    def binarise_data(self, ys):
        median = torch.median(ys[:, :self.num_rows], dim=1, keepdim=True)[0]
        ys = (ys > median)

        if self.flip:
            # Random flip to balance dataset
            flip = torch.randint(0, 2, (self.bs, 1))
            ys = torch.logical_xor(ys, flip)

        return ys.long()

    def __len__(self):
        num_batches = self.len // ((self.num_rows + self.num_target) * self.bs)
        return num_batches


# Dataset2vec requires different dataloader from GNN. Returns all pairs of x and y.
def d2v_pairer(xs, ys):
    #    # torch.Size([2, 5, 10]) torch.Size([2, 5, 1])
    bs, num_rows, num_xs = xs.shape

    xs = xs.view(bs * num_rows, num_xs)
    ys = ys.view(bs * num_rows, 1)

    pair_flat = torch.empty(bs * num_rows, num_xs, 2, device=xs.device)
    for k, (xs_k, ys_k) in enumerate(zip(xs, ys)):
        # Only allow 1D for ys
        ys_k = ys_k.repeat(num_xs)
        pairs = torch.stack([xs_k, ys_k], dim=-1)

        pair_flat[k] = pairs

    pairs = pair_flat.view(bs, num_rows, num_xs, 2)

    return pairs


if __name__ == "__main__":
    # Test if dataloader works.
    np.random.seed(0)
    torch.manual_seed(0)

    dl = AdultDataLoader(bs=2, num_rows=10, num_target=3)
    #
    # for i in range(15):
    #     num_unique = np.unique(dl.data[:, i])
    #     print(i, len(num_unique))

    means = []
    for x, y in dl:
        y_mean = torch.mean(y, dtype=float)
        means.append(y_mean)

    means = torch.stack(means)
    means = torch.mean(means)
    print(means)
