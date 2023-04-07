import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataloader import AdultDataLoader, d2v_pairer
from data_generation import MLPDataLoader
from GAtt_Func import GATConvFunc
from save_holder import SaveHolder

from config import get_config


# Dataset2vec model
class SetSetModel(nn.Module):
    def __init__(self, h_size, out_size, pos_enc_dim, model_depths, reparam_weight, reparam_pos_enc):
        super().__init__()
        self.reparam_weight = reparam_weight
        self.reparam_pos_enc = reparam_pos_enc

        f_depth, g_depth, h_depth, pos_depth = model_depths

        self.relu = nn.ReLU()

        # f network
        self.f_in = nn.Linear(2, h_size)
        self.fs = nn.ModuleList([])
        for _ in range(f_depth - 2):
            self.fs.append(nn.Linear(h_size, h_size))
        self.f_out = nn.Linear(h_size, h_size)

        # g network
        self.gs = nn.ModuleList([])
        for _ in range(g_depth):
            self.gs.append(nn.Linear(h_size, h_size))

        # h network
        self.h_in = nn.Linear(h_size, h_size)
        self.hs = nn.ModuleList([])
        for _ in range(h_depth - 2):
            self.hs.append(nn.Linear(h_size, h_size))
        self.h_out = nn.Linear(h_size, out_size)
        if reparam_weight:
            self.h_out_lvar = nn.Linear(h_size, out_size)

        # Embedding Network
        self.p_in = nn.Linear(h_size, h_size)
        self.ps = nn.ModuleList([])
        for _ in range(pos_depth - 2):
            self.ps.append(nn.Linear(h_size, h_size))
        self.p_out = nn.Linear(h_size, pos_enc_dim)
        if self.reparam_pos_enc:
            self.p_out_lvar = nn.Linear(h_size, pos_enc_dim)

    def forward_layers(self, x):
        # x.shape = [num_rows, num_cols, 2]

        # f network
        x = self.relu(self.f_in(x))  # [num_rows, num_cols, h_size]
        for layer in self.fs:
            x_r = self.relu(layer(x))
            x = x + x_r
        x = self.relu(self.f_out(x))

        x = torch.mean(x, dim=0)  # [num_rows, y_dim, h_size]
        x_save = x

        # g network
        for layer in self.gs:
            x = self.relu(layer(x))
        x = torch.mean(x, dim=0)  # [h_size]

        # h network
        x = self.relu(self.h_in(x))
        for layer in self.hs:
            x_r = self.relu(layer(x))
            x = x + x_r
        x_out = self.h_out(x)
        if self.reparam_weight:
            x_lvar = self.h_out_lvar(x)
            x_out = torch.stack([x_out, x_lvar])

        # Positional Encoding
        pos_enc = self.relu(self.p_in(x_save))
        for layer in self.ps:
            pos_enc = self.relu(layer(pos_enc))
        pos_out = self.p_out(pos_enc)
        if self.reparam_pos_enc:
            pos_lvar = self.p_out_lvar(pos_enc)
            pos_out = torch.stack([pos_out, pos_lvar])

        return x_out, pos_out

    def forward(self, xs):
        # xs.shape = [BS][x_dim, n_samples, 2]
        # Final dim of x is pair (x_i, y).

        # TODO: Properly handle batches
        ys, pos_encs = [], []
        for x in xs:
            outs, embeds = self.forward_layers(x)
            ys.append(outs)
            pos_encs.append(embeds)

        ys = torch.stack(ys)
        pos_encs = torch.stack(pos_encs)

        return ys, pos_encs


# Generates weights from dataset2vec model outputs.
class WeightGenerator(nn.Module):
    def __init__(self, in_dim, hid_dim, out_sizes: list, gen_layers, reparam_weight, reparam_pos_enc):
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
        self.gen_layers = gen_layers

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

        module = [nn.Linear(self.gen_in_dim, self.gen_hid_dim), nn.ReLU()]
        for _ in range(self.gen_layers - 2):
            module.append(nn.Linear(self.gen_hid_dim, self.gen_hid_dim))
            module.append(nn.ReLU())
        module.append(nn.Linear(self.gen_hid_dim, tot_params))

        module = nn.Sequential(*module)

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

        reparam_weight = cfg["reparam_weight"]
        reparam_pos_enc = cfg["reparam_pos_enc"]
        self.reparam_weight = reparam_weight
        self.reparam_pos_enc = reparam_pos_enc

        set_h_dim = cfg["set_h_dim"]
        set_out_dim = cfg["set_out_dim"]
        pos_enc_dim = cfg["pos_enc_dim"]

        weight_hid_dim = cfg["weight_hid_dim"]

        gat_heads = cfg["gat_heads"]
        gat_hid_dim = cfg["gat_hid_dim"]
        gat_in_dim = cfg["gat_in_dim"]
        gat_out_dim = cfg["gat_out_dim"]

        d2v_layers = cfg["d2v_layers"]
        gen_layers = cfg["gen_layers"]
        gat_layers = cfg["gat_layers"]

        gat_shapes = [(gat_in_dim, gat_hid_dim, gat_heads)] + [(gat_hid_dim, gat_hid_dim, gat_heads) for _ in range(gat_layers - 2)] + [
            (gat_hid_dim, gat_out_dim, gat_heads)]

        self.d2v_model = SetSetModel(h_size=set_h_dim, out_size=set_out_dim,
                                     pos_enc_dim=pos_enc_dim, model_depths=d2v_layers,
                                     reparam_weight=reparam_weight, reparam_pos_enc=reparam_pos_enc)
        self.weight_model = WeightGenerator(in_dim=set_out_dim, hid_dim=weight_hid_dim,
                                            out_sizes=gat_shapes, gen_layers=gen_layers,
                                            reparam_weight=reparam_weight, reparam_pos_enc=reparam_pos_enc)
        self.gnn_model = GNN()

    # Forward Meta set and train
    def forward_meta(self, pairs_meta):
        embed_meta, pos_enc = self.d2v_model(pairs_meta)

        # Reparametrisation trick. Keep copies of log_var here
        if self.reparam_weight:
            embed_means, embed_lvar = embed_meta[:, 0], embed_meta[:, 1]
            std = torch.exp(0.5 * embed_lvar)
            eps = torch.randn_like(std)
            embed_meta = embed_means + eps * std

            self.embed_lvar = embed_lvar
            self.embed_means = embed_means

        if self.reparam_pos_enc:
            pos_means, pos_lvar = pos_enc[:, 0], pos_enc[:, 1]
            std = torch.exp(0.5 * pos_lvar)
            eps = torch.randn_like(std)
            pos_enc = pos_means + eps * std

            self.pos_lvar = pos_lvar
            self.pos_means = pos_means

        return embed_meta, pos_enc

    def forward_target(self, xs_target, embed_meta, pos_enc):
        weights_target = self.weight_model(embed_meta)
        preds_meta = self.gnn_model(xs_target, pos_enc, weights_target)

        return preds_meta

    def loss_fn(self, preds, targs):
        cross_entropy = torch.nn.functional.cross_entropy(preds, targs)

        kl_div = 0
        if self.reparam_weight:
            div = 1 + self.embed_lvar - self.embed_means.square() - self.embed_lvar.exp()  # [BS, embed_dim]
            kl_div += torch.mean(-0.5 * torch.sum(div, dim=-1))

        if self.reparam_pos_enc:
            div = 1 + self.pos_lvar - self.pos_means.square() - self.pos_lvar.exp()  # [BS, num_cols, emb_dim]
            kl_div += torch.mean(-0.5 * torch.sum(div, dim=-1))

        loss = cross_entropy + 0.1 * kl_div
        return loss

        # kl_div = -0.5 * torch.sum()


def train():
    save_holder = SaveHolder(".")

    all_cfgs = get_config()
    cfg = all_cfgs["Optim"]
    lr = cfg["lr"]

    cfg = all_cfgs["DL_params"]
    bs = cfg["bs"]
    num_rows = cfg["num_rows"]
    num_targets = cfg["num_targets"]
    flip = cfg["flip"]

    cfg = all_cfgs["Settings"]
    num_epochs = cfg["num_epochs"]
    print_interval = cfg["print_interval"]
    save_batch = cfg["save_batch"]

    # train_dl = AdultDataLoader(bs=bs, num_rows=num_rows, num_target=num_targets, flip=flip, split="train")
    # val_dl = AdultDataLoader(bs=16, num_rows=num_rows, num_target=1, flip=flip, split="val")
    train_dl = MLPDataLoader(bs=bs, num_rows=num_rows, num_target=num_targets, num_cols=4, config=all_cfgs["MLP_DL_params"])

    model = ModelHolder()

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    accs, losses = [], []
    # val_accs, val_losses = [], []
    batch_no = 0
    for epoch in range(num_epochs):
        # Train loop
        model.train()
        for xs, ys in train_dl:
            batch_no += 1

            # xs.shape = [bs, num_rows, num_xs]
            xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
            ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
            # Splicing like this changes the tensor's stride. Fix here:
            xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
            ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
            ys_target = ys_target.view(-1)

            # Reshape for dataset2vec
            pairs_meta = d2v_pairer(xs_meta, ys_meta)

            # First pass with the meta-set, train d2v and get embedding.
            embed_meta, pos_enc = model.forward_meta(pairs_meta)
            # Second pass using previous embedding and train weight encoder
            # During testing, rows of the dataset don't interact.
            ys_pred_targ = model.forward_target(xs_target, embed_meta, pos_enc).view(-1, 2)

            loss = model.loss_fn(ys_pred_targ, ys_target)
            loss.backward()
            optim.step()
            optim.zero_grad()

            # Accuracy recording
            predicted_labels = torch.argmax(ys_pred_targ, dim=1)
            accuracy = (predicted_labels == ys_target).sum().item() / len(ys_target)

            accs.append(accuracy), losses.append(loss.item())

            if batch_no % print_interval == 0:
                print()
                print(f'{epoch=}, {batch_no=}')
                print("Targets:    ", ys_target.numpy())
                print("Predictions:", predicted_labels.numpy())
                print(f'Mean accuracy: {np.mean(accs[-print_interval:]) * 100:.2f}')
                print(torch.mean(model.pos_lvar).item(), torch.mean(model.pos_means).item())

            # # val_loop
            # model.eval()
            # epoch_accs, epoch_losses = [], []
            # for batch_no, (xs, ys) in enumerate(val_dl):
            #     # xs.shape = [bs, num_rows, num_xs]
            #
            #     xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
            #     ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
            #     # Splicing like this changes the tensor's stride. Fix here:
            #     xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
            #     ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
            #     ys_target = ys_target.view(-1)
            #     # Reshape for dataset2vec
            #     pairs_meta = d2v_pairer(xs_meta, ys_meta)
            #
            #     with torch.no_grad():
            #         embed_meta, pos_enc = model.forward_meta(pairs_meta)
            #         ys_pred_targ = model.forward_target(xs_target, embed_meta, pos_enc).view(-1, 2)
            #
            #     loss = torch.nn.functional.cross_entropy(ys_pred_targ, ys_target)
            #
            #     # Accuracy recording
            #     predicted_labels = torch.argmax(ys_pred_targ, dim=1)
            #     accuracy = (predicted_labels == ys_target).sum().item() / len(ys_target)
            #
            #     epoch_accs.append(accuracy), epoch_losses.append(loss.item())
            #
            # print()
            # print(f'Validation Stats: ')
            # print(f'Accuracy: {np.mean(epoch_accs) * 100:.2f}, Loss: {np.mean(epoch_losses) :.4g}')
            # val_accs.append(epoch_accs)
            # val_losses.append(epoch_losses)

            if batch_no % save_batch == 0:
                save_holder.save_model(model)
                history = {"accs": accs, "loss": losses}
                save_holder.save_history(history)


if __name__ == "__main__":
    import random

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    train()
