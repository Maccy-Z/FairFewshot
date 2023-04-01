import torch
import torch.nn as nn
from dataloader import AdultDataLoader
import numpy as np
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
        for gat_shape in out_sizes:
            self.layer_model.append(self.gen_layer(*gat_shape))

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
    def __init__(self):
        super().__init__()
        self.GATConv = GATConvFunc()

    # Generate additional fixed embeddings / graph
    def graph_and_embeddings(self, in_nodes):
        # Densely connected graph
        nodes = torch.arange(in_nodes)
        interleave = torch.repeat_interleave(nodes, in_nodes)
        repeat = torch.tile(nodes, (in_nodes,))
        edge_idx = torch.stack([interleave, repeat])

        # Positional embeddings
        x_pos = torch.arange(in_nodes).to(torch.float32)
        x_pos = x_pos - in_nodes / 2

        return edge_idx, x_pos

    def forward(self, x, weight_list: list[list[torch.Tensor]]):
        """
        :param x: x.shape = [BS, num_rows, num_xs]
        :param weight_list: weight_list.shape = [BS, num_layers, tensor[4]]
        :return:
        """

        x = x[0, 0]
        edge_idx, x_pos_embed = self.graph_and_embeddings(x.shape[0])

        x = torch.stack([x, x_pos_embed], dim=1)


        weight_list = weight_list[0]

        for layer_weights in weight_list:
            lin_weight, src_weight, dst_weight, bias_weight = layer_weights
            print(x.shape)

            x = self.GATConv(x, edge_idx, lin_weight, src_weight, dst_weight, bias_weight)
            print("Single loop completed")



def train():
    dl = AdultDataLoader(bs=3, num_rows=5, num_xs=10)

    d2v_model = SetSetModel(h_size=64, out_size=64)

    heads = 2
    hid_dim = 16
    in_dim = 2
    weight_model = WeightGenerator(in_dim=64, hid_dim=64, out_sizes=[(in_dim, hid_dim, heads), (hid_dim, 2, heads)])
    gnn_model = GNN()

    print()

    for xs, ys in dl:
        # xs.shape = [bs, num_rows, num_xs]
        # print(xs.shape)
        pairs = dl.dataset_2_vec_dl(xs, ys)
        # print(pairs.shape)
        output = d2v_model(pairs)

        weights = weight_model(output)

        gnn_model(x, weights)

        exit(10)
    # print(output.shape)


if __name__ == "__main__":
    train()
