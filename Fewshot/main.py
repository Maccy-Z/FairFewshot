import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataloader import AdultDataLoader, d2v_pairer
from GAtt_Func import GATConvFunc

from config import get_config


# Dataset2vec model
class SetSetModel(nn.Module):
    def __init__(self, h_size, out_size, pos_enc_dim):
        super().__init__()
        self.h_size = h_size
        self.out_size = out_size

        # f network
        self.f_1 = nn.Linear(2, h_size)
        self.f_2_r = nn.Linear(h_size, h_size)
        self.f_3_r = nn.Linear(h_size, h_size)
        # self.f_4_r = nn.Linear(h_size, h_size)
        self.f_5 = nn.Linear(h_size, h_size)

        # g network
        self.g_1 = nn.Linear(h_size, h_size)
        self.g_2 = nn.Linear(h_size, h_size)

        # h network
        self.h_1 = nn.Linear(h_size, h_size)
        self.h_2_r = nn.Linear(h_size, h_size)
        self.h_3_r = nn.Linear(h_size, h_size)
        # self.h_4_r = nn.Linear(h_size, h_size)
        self.h_5 = nn.Linear(h_size, out_size)

        # Embedding Network
        self.p_1 = nn.Linear(h_size, h_size)
        self.p_2 = nn.Linear(h_size, pos_enc_dim)

        self.relu = nn.ReLU()

    def forward(self, xs):
        # xs.shape = [BS][x_dim, n_samples, 2]
        # Final dim of x is pair (x_i, y)

        # TODO: Properly handle batches
        ys, pos_encs = [], []
        for x in xs:
            outs, embeds = self.forward_layers(x)
            ys.append(outs)
            pos_encs.append(embeds)

        ys = torch.stack(ys)
        pos_encs = torch.stack(pos_encs)

        return ys, pos_encs

    def forward_layers(self, x):
        # x.shape = [num_rows, num_cols, 2]

        # f network
        x = self.relu(self.f_1(x))  # [num_rows, num_cols, h_size]
        x_r = self.relu(self.f_2_r(x))
        x = x + x_r
        # x_r = self.relu(self.f_3_r(x))
        # x = x + x_r
        # x_r = self.relu(self.f_4_r(x))
        # x = x + x_r
        x = self.relu(self.f_5(x))

        x = torch.mean(x, dim=0)  # [num_rows, y_dim, h_size]
        x_save = x

        # g network
        x = self.relu(self.g_1(x))
        x = self.relu(self.g_2(x))
        x = torch.mean(x, dim=0)  # [h_size]

        # h network
        x = self.relu(self.h_1(x))
        x_r = self.relu(self.h_2_r(x))
        x = x + x_r
        # x_r = self.relu(self.h_3_r(x))
        # x = x + x_r
        # x_r = self.relu(self.h_4_r(x))
        # x = x + x_r
        x = self.relu(self.h_5(x))

        # Positional Encoding
        pos_enc = self.p_1(x_save)
        pos_enc = self.relu(pos_enc)
        pos_enc = self.p_2(pos_enc)

        return x, pos_enc


# Generates weights from dataset2vec model outputs.
class WeightGenerator(nn.Module):
    def __init__(self, in_dim, hid_dim, out_sizes: list):
        """
        :param in_dim: Dim of input from dataset2vec
        :param hid_dim: Internal hidden size
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

        # Weights for final linear classificaion layer
        self.num_classes = 2
        self.gat_out_dim = self.out_sizes[-1][-2]
        lin_out_dim = self.gat_out_dim * self.num_classes
        self.weight_gen_linear = nn.Sequential(
            nn.Linear(self.gen_in_dim, self.gen_hid_dim),
            nn.ReLU(),
            nn.Linear(self.gen_hid_dim, lin_out_dim)
        )

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

    def forward(self, d2v_embed):
        # d2v_embed.shape = [BS, d2v_out]

        layer_weights = []
        for module, split_idxs, (gat_in, gat_out, gat_heads) in self.layer_model:

            all_weights = module(d2v_embed)  # [BS, layers_size]

            # Split weights into individual matrices and a list of batches.
            batch = []
            for batch_weights in all_weights:
                lin, src, dst, bias = torch.split(batch_weights, split_idxs)
                lin, src, dst, bias = F.normalize(lin, dim=0), F.normalize(src, dim=0), F.normalize(dst, dim=0), F.normalize(bias, dim=0)
                # Reshape each weight matrix
                lin = lin.view(gat_out * gat_heads, gat_in)
                src = src.view(1, gat_heads, gat_out)
                dst = dst.view(1, gat_heads, gat_out)
                bias = bias.view(gat_out * gat_heads)

                batch.append((lin, src, dst, bias))
            layer_weights.append(batch)

        layer_weights = list(zip(*layer_weights))  # [BS, num_layers, tensor[4]]

        # Weights for linear layer
        lin_weights = self.weight_gen_linear(d2v_embed)
        lin_weights = lin_weights.view(-1, self.num_classes, self.gat_out_dim)

        return layer_weights, lin_weights


class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.GATConv = GATConvFunc()

    # Generate additional fixed embeddings / graph
    def graph_matrix(self, num_rows, num_xs):
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

        return edge_idx

    def forward(self, xs, pos_enc, weight_list: tuple[list[list[torch.Tensor]], list]):
        """
        :param xs:              shape = [BS, num_rows, num_xs]
        :param pos_enc:         shape = [BS, num_xs, enc_dim]
        :param weight_list:     shape[0] = [BS, num_layers, tensor[4]]
                                shape[1] = [BS, tensor[1]]

        :return output:         shape = [BS, num_rows, num_xs, 2]
        """
        bs, num_rows, num_cols = xs.shape
        gat_weights, lin_weights = weight_list

        # Flatten each table
        pos_enc = pos_enc.unsqueeze(1).repeat(1, num_rows, 1, 1).view(bs, num_rows * num_cols, -1)
        xs = xs.view(bs, num_rows * num_cols, 1)
        xs = torch.cat([xs, pos_enc], dim=-1)

        edge_idx = self.graph_matrix(num_rows, num_cols)

        output = []
        for batch_weights, final_weight, x in zip(gat_weights, lin_weights, xs):

            # Forward a single batch
            for layer_weights in batch_weights:
                lin_weight, src_weight, dst_weight, bias_weight = layer_weights

                x = self.GATConv(x, edge_idx, lin_weight, src_weight, dst_weight, bias_weight)

            # Sum GAT node outputs for final predictions.
            x = x.view(num_rows, num_cols, -1)
            x = x.sum(-2)

            # Final linear classification layer
            x = F.linear(x, final_weight)
            output.append(x)

        output = torch.stack(output)
        return output


class ModelHolder(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = get_config()["NN_dims"]

        pos_enc_dim = cfg["pos_enc_dim"]
        set_h_dim = cfg["set_h_dim"]
        set_out_dim = cfg["set_out_dim"]
        weight_hid_dim = cfg["weight_hid_dim"]
        gat_heads = cfg["gat_heads"]
        gat_hid_dim = cfg["gat_hid_dim"]
        gat_in_dim = cfg["gat_in_dim"]
        gat_out_dim = cfg["gat_out_dim"]

        self.d2v_model = SetSetModel(h_size=set_h_dim, out_size=set_out_dim, pos_enc_dim=pos_enc_dim)
        self.weight_model = WeightGenerator(in_dim=set_out_dim, hid_dim=weight_hid_dim,
                                            out_sizes=[(gat_in_dim, gat_hid_dim, gat_heads),
                                                       (gat_hid_dim, gat_hid_dim, gat_heads),
                                                       (gat_hid_dim, gat_out_dim, gat_heads)])
        self.gnn_model = GNN()

    # Forward Meta set and train
    def forward_meta(self, xs_meta, pairs_meta):
        embed_meta, pos_enc = self.d2v_model(pairs_meta)
        weights_meta = self.weight_model(embed_meta)
        preds_meta = self.gnn_model(xs_meta, pos_enc, weights_meta)

        return preds_meta, embed_meta.detach(), pos_enc.detach()

    def forward_target(self, xs_target, embed_meta, pos_enc):
        weights_target = self.weight_model(embed_meta)
        preds_meta = self.gnn_model(xs_target, pos_enc, weights_target)

        return preds_meta


def train():
    cfg = get_config()["LR"]
    meta_lr = cfg["meta_lr"]
    target_lr = cfg["target_lr"]

    cfg = get_config()["DL_params"]
    bs = cfg["bs"]
    num_rows = cfg["num_rows"]
    num_targets = cfg["num_targets"]
    flip = cfg["flip"]

    dl = AdultDataLoader(bs=bs, num_rows=num_rows, num_target=num_targets, flip=flip, split="train")

    model = ModelHolder()

    optim_meta = torch.optim.Adam(model.d2v_model.parameters(), lr=meta_lr)
    optim_target = torch.optim.Adam(model.weight_model.parameters(), lr=target_lr)

    for epoch in range(100000):
        accs = []

        for batch_no, (xs, ys) in enumerate(dl):
            # xs.shape = [bs, num_rows, num_xs]

            xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
            ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
            # Splicing like this changes the tensor's stride. Fix here:
            xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
            ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()

            # Reshape for dataset2vec
            pairs_meta = d2v_pairer(xs_meta, ys_meta)

            # First pass with the meta-set, train d2v and get embedding.
            ys_pred_meta, embed_meta, pos_enc = model.forward_meta(xs_meta, pairs_meta)

            ys_meta = ys_meta.view(-1)
            ys_pred_meta = ys_pred_meta.view(-1, 2)

            loss_meta = torch.nn.functional.cross_entropy(ys_pred_meta, ys_meta)
            loss_meta.backward()
            model.weight_model.zero_grad()

            # Second pass using previous embedding and train weight encoder
            # During testing, rows of the dataset don't interact.
            ys_pred_targ = model.forward_target(xs_target, embed_meta, pos_enc)

            ys_target = ys_target.view(-1)
            ys_pred_targ = ys_pred_targ.view(-1, 2)

            loss_target = torch.nn.functional.cross_entropy(ys_pred_targ, ys_target)
            loss_target.backward()
            optim_target.step()
            optim_meta.step()
            optim_target.zero_grad()
            optim_meta.zero_grad()

            # Accuracy recording
            predicted_labels = torch.argmax(ys_pred_targ, dim=1)
            accuracy = (predicted_labels == ys_target).sum().item() / len(ys_target)
            accs.append(accuracy)

            if batch_no % 100 == 0:
                print()
                print(f'{epoch=}, {batch_no=}')
                print("Targets:    ", ys_target.numpy())
                print("Predictions:", predicted_labels.numpy())
                print(f'Mean accuracy: {np.mean(accs) * 100:.2f}')
                accs = []


if __name__ == "__main__":
    np.random.seed(1)
    torch.manual_seed(1)
    train()
