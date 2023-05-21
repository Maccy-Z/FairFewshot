# Loads datasets.
import torch

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


