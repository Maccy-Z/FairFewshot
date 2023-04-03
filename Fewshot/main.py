import torch
import torch.nn as nn
import numpy as np

from dataloader import AdultDataLoader, d2v_pairer
from GAtt_Func import GATConvFunc


# Dataset2vec model
class SetSetModel(nn.Module):
    def __init__(self, h_size, out_size):
        super().__init__()
        self.h_size = h_size
        self.out_size = out_size

        # f network
        self.f_1 = nn.Linear(2, h_size)
        self.f_2_r = nn.Linear(h_size, h_size)
        self.f_3_r = nn.Linear(h_size, h_size)
        self.f_4_r = nn.Linear(h_size, h_size)
        self.f_5 = nn.Linear(h_size, h_size)

        # g network
        self.g_1 = nn.Linear(h_size, h_size)
        self.g_2 = nn.Linear(h_size, h_size)

        # h network
        self.h_1 = nn.Linear(h_size, h_size)
        self.h_2_r = nn.Linear(h_size, h_size)
        self.h_3_r = nn.Linear(h_size, h_size)
        self.h_4_r = nn.Linear(h_size, h_size)
        self.h_5 = nn.Linear(h_size, out_size)

        self.relu = nn.ReLU()

    def forward(self, xs):
        # xs.shape = [BS][x_dim, n_samples, 2]
        # Final dim of x is pair (x_i, y)

        # TODO: Properly handle batches
        ys = []
        for x in xs:
            ys.append(self.forward_layers(x))

        ys = torch.stack(ys)

        return ys

    def forward_layers(self, x):
        # f network
        # x.shape = [x_dim, n_samples, 2]
        x = self.relu(self.f_1(x))  # [x_dim, n_samples, h_size]
        x_r = self.relu(self.f_2_r(x))
        x = x + x_r
        x_r = self.relu(self.f_3_r(x))
        x = x + x_r
        x_r = self.relu(self.f_4_r(x))
        x = x + x_r
        x = self.relu(self.f_5(x))

        x = torch.mean(x, dim=-2)  # [x_dim, y_dim, h_size]

        # g network
        x = self.relu(self.g_1(x))
        x = self.relu(self.g_2(x))
        x = torch.mean(x, dim=0)  # [h_size]

        # h network
        x = self.relu(self.h_1(x))
        x_r = self.relu(self.h_2_r(x))
        x = x + x_r
        x_r = self.relu(self.h_3_r(x))
        x = x + x_r
        x_r = self.relu(self.h_4_r(x))
        x = x + x_r
        x = self.relu(self.h_5(x))

        return x


# Generates weights from dataset2vec model outputs.
class WeightGenerator(nn.Module):
    def __init__(self, in_dim, hid_dim, out_sizes: list):
        """
        :param in_size: Dim of input from dataset2vec
        :param out_sizes: List of params for GATConv layers [num_layers][in_dim, out_dim, heads]
            Each GATConv layer requires:
                linear_weight = (out_dim * heads, in_dim)
                src = (1, heads, out_dim)
                dst = (1, heads, out_dim)
                bias = (out_dim * heads)
        """
        super().__init__()
        self.gen_in_dim = in_dim
        self.gen_hid_dim = hid_dim
        self.out_sizes = out_sizes

        # Each GAT layer has a submodel to generate weights
        self.layer_model = []
        for i, gat_shape in enumerate(out_sizes):
            model, weights, dims = self.gen_layer(*gat_shape)
            self.layer_model.append([model, weights, dims])

            # Need to register each module manually
            self.add_module(f'weight_gen_{i}', model)

    def gen_layer(self, gat_in_dim, gat_out_dim, gat_heads):
        # WARN: GAT output size is heads * out_dim, so correct here.
        gat_out_dim = gat_out_dim // gat_heads

        # Number of parameters model needs to output:
        lin_weight_params = gat_out_dim * gat_heads * gat_in_dim
        src_params = gat_heads * gat_out_dim
        dst_params = gat_heads * gat_out_dim
        bias_params = gat_out_dim * gat_heads

        tot_params = lin_weight_params + src_params + dst_params + bias_params
        # Save indices of weights
        weight_idxs = [lin_weight_params, src_params, dst_params, bias_params]

        module = nn.Sequential(
            nn.Linear(self.gen_in_dim, self.gen_hid_dim),
            nn.ReLU(),
            nn.Linear(self.gen_hid_dim, tot_params)
        )

        return module, weight_idxs, (gat_in_dim, gat_out_dim, gat_heads)

    def forward(self, d2v_xs):
        # d2v_xs.shape = [BS, d2v_out]

        layer_weights = []
        for module, split_idxs, (gat_in, gat_out, gat_heads) in self.layer_model:

            all_weights = module(d2v_xs)  # [BS, layers_size]

            # Split weights into individual matrices and a list of batches.
            batch = []
            for batch_weights in all_weights:
                lin, src, dst, bias = torch.split(batch_weights, split_idxs)

                # Reshape each weight matrix
                lin = lin.view(gat_out * gat_heads, gat_in)
                src = src.view(1, gat_heads, gat_out)
                dst = dst.view(1, gat_heads, gat_out)
                bias = bias.view(gat_out * gat_heads)

                batch.append((lin, src, dst, bias))
            layer_weights.append(batch)

        layer_weights = list(zip(*layer_weights))

        # layer_weights.shape = [BS, num_layers, tensor[4]]
        return layer_weights


class GNN(nn.Module):
    def __init__(self, gat_out_dim):
        super().__init__()
        self.GATConv = GATConvFunc()
        self.out_lin = nn.Linear(gat_out_dim, 2)

    # Generate additional fixed embeddings / graph
    def graph_and_embeddings(self, num_rows, num_xs):
        # Densely connected graph
        nodes = torch.arange(num_xs)
        interleave = torch.repeat_interleave(nodes, num_xs)
        repeat = torch.tile(nodes, (num_xs,))
        base_edge_idx = torch.stack([interleave, repeat])

        # Repeat edge_index over num_rows to get block diagonal adjacency matrix by adding num_xs to everything
        edge_idx = []
        for i in np.arange(num_rows):
            edge_idx.append(base_edge_idx + i * num_xs)

        edge_idx = torch.cat(edge_idx, dim=-1)

        # Positional embeddings
        x_pos = torch.zeros(num_xs * num_rows).view(-1, 1)

        return edge_idx, x_pos

    def forward(self, xs, weight_list: list[list[torch.Tensor]]):
        """
        :param xs:               x.shape = [BS, num_rows, num_xs]
        :param weight_list:     weight_list.shape = [BS, num_layers, tensor[4]]

        :return output:         output.shape = [BS, num_rows, num_xs, 2]
        """

        bs, num_rows, num_xs = xs.shape
        # Flatten each table
        xs = xs.view(bs, -1)

        edge_idx, x_pos_embed = self.graph_and_embeddings(num_rows, num_xs)

        # TODO: positional embedding.
        x_pos_embed = torch.zeros_like(xs)

        xs = torch.stack([xs, x_pos_embed], dim=-1)

        output = []
        for batch_weights, x in zip(weight_list, xs):

            # Forward a single batch
            for layer_weights in batch_weights:
                lin_weight, src_weight, dst_weight, bias_weight = layer_weights

                x = self.GATConv(x, edge_idx, lin_weight, src_weight, dst_weight, bias_weight)

            # Sum GAT node outputs for final predictions.
            x = x.view(num_rows, num_xs, -1)
            x = x.sum(-2)

            output.append(x)

        output = torch.stack(output)
        output = output.view(bs, num_rows, -1)

        output = self.out_lin(output)

        return output


class ModelHolder(nn.Module):
    def __init__(self):
        super().__init__()

        set_h_dim = 64
        set_out_dim = 32
        weight_hid_dim = 64
        gat_heads = 2
        gat_hid_dim = 64
        gat_in_dim = 2
        gat_out_dim = 16

        self.d2v_model = SetSetModel(h_size=set_h_dim, out_size=set_out_dim)
        self.weight_model = WeightGenerator(in_dim=set_out_dim, hid_dim=weight_hid_dim, out_sizes=[(gat_in_dim, gat_hid_dim, gat_heads), (gat_hid_dim, gat_out_dim, gat_heads)])
        self.gnn_model = GNN(gat_out_dim)

    def forward(self, xs, pairs):
        output = self.d2v_model(pairs)
        weights = self.weight_model(output)
        preds = self.gnn_model(xs, weights)

        return preds

    def meta_params(self):
        return self.d2v_model.parameters()

    def target_params(self):
        return list(self.weight_model.parameters()) + list(self.gnn_model.parameters())


def train():
    dl_train = AdultDataLoader(bs=3, num_rows=5, split="train")
    dl_valid = AdultDataLoader(bs=3, num_rows=5, split="val")

    model = ModelHolder()

    optim_meta = torch.optim.SGD(model.meta_params(), lr=0.001, momentum=0.9)
    optim_target = torch.optim.SGD(model.target_params(), lr=0.001, momentum=0.9)

    accs = []
    for batch_no, ((xs_meta, ys_meta), (xs_target, ys_target)) in enumerate(zip(dl_train, dl_valid)):
        # xs.shape = [bs, num_rows, num_xs]

        # Temp target
        label = torch.randint(0, 2, (1,))
        ys_meta = torch.ones_like(ys_meta) * label
        xs_meta = torch.ones_like(xs_meta) * label

        label = torch.randint(0, 2, (1,))
        ys_target = torch.ones_like(ys_target) * label
        xs_target = torch.ones_like(xs_target) * label

        # Reshape for dataset2vec
        pairs_meta = d2v_pairer(xs_meta, ys_meta)
        pairs_target = d2v_pairer(xs_target, ys_target)

        # One pass with the meta-set and one pass with target-set.

        # Meta pass
        ys_pred_meta = model(xs_meta, pairs_meta)

        ys_meta = ys_meta.view(-1)
        ys_pred_meta = ys_pred_meta.view(-1, 2)

        loss = torch.nn.functional.cross_entropy(ys_pred_meta, ys_meta)

        loss.backward()
        optim_meta.step()
        optim_meta.zero_grad()
        optim_target.zero_grad()

        # Target pass
        ys_pred_targ = model(xs_target, pairs_target)

        ys_target = ys_target.view(-1)
        ys_pred_targ = ys_pred_targ.view(-1, 2)

        loss = torch.nn.functional.cross_entropy(ys_pred_targ, ys_target)

        loss.backward()
        optim_target.step()
        optim_target.zero_grad()
        optim_meta.zero_grad()

        # Accuracy recording
        predicted_labels = torch.argmax(ys_pred_targ, dim=1)
        accuracy = (predicted_labels == ys_target).sum().item() / len(ys_target)
        accs.append(accuracy)

        if batch_no % 50 == 0:
            print(f'{np.mean(accs)*100:.2f}')
            accs = []


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    train()
