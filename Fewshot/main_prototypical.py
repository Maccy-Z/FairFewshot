import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import numpy as np
import itertools
import time

from GAtt_Func import GATConvFunc
from save_holder import SaveHolder
from config import get_config, Config
from AllDataloader import SplitDataloader, d2v_pairer
from torch.optim.lr_scheduler import StepLR

cfg2 = Config()

class ResBlock(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_blocks, out_relu=True):
        super().__init__()
        self.out_relu = out_relu

        self.res_modules = nn.ModuleList([])
        self.lin_in = nn.Linear(in_size, hid_size)

        for _ in range(n_blocks - 2):
            self.res_modules.append(nn.Linear(hid_size, hid_size))

        self.lin_out = nn.Linear(hid_size, out_size)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_in(x)
        for layer in self.res_modules:
            x_r = self.act(layer(x))
            x = x + x_r

        if self.out_relu:
            out = self.act(self.lin_out(x))
        else:
            out = self.lin_out(x)
        return out, x


# Dataset2vec model
class SetSetModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.reparam_weight = cfg["reparam_weight"]
        self.reparam_pos_enc = cfg["reparam_pos_enc"]

        h_size = cfg["set_h_dim"]
        out_size = cfg["set_out_dim"]

        pos_enc_bias = cfg["pos_enc_bias"]
        pos_enc_dim = cfg["pos_enc_dim"]
        pos_depth = cfg["pos_depth"]

        model_depths = cfg["d2v_layers"]
        f_depth, g_depth, h_depth = model_depths

        self.relu = nn.ReLU()

        # f network
        self.fs = ResBlock(2, h_size, h_size, n_blocks=f_depth)

        # g network
        self.gs = ResBlock(h_size, h_size, h_size, n_blocks=g_depth)

        # h network
        self.hs = ResBlock(h_size, h_size, out_size, n_blocks=h_depth, out_relu=False)

        if self.reparam_weight:
            self.h_out_lvar = nn.Linear(h_size, out_size)

        # Positional embedding Network
        self.ps = nn.ModuleList([])
        for _ in range(pos_depth - 1):
            self.ps.append(nn.Linear(h_size, h_size))

        self.p_out = nn.Linear(h_size, pos_enc_dim, bias=(pos_enc_bias != "off"))
        if self.reparam_pos_enc:
            self.p_out_lvar = nn.Linear(h_size, pos_enc_dim)

        if pos_enc_bias == "zero":
            # print(f'Positional encoding bias init to 0')
            self.p_out.bias.data.fill_(0)

    def forward_layers(self, x):
        # x.shape = [num_rows, num_cols, 2]

        # f network
        x, _ = self.fs(x)  # [num_rows, num_cols, h_size]

        x = torch.mean(x, dim=0)  # [num_rows, y_dim, h_size]
        x_pos_enc = x

        # g network
        x, _ = self.gs(x)
        x = torch.mean(x, dim=0)  # [h_size]

        # h network
        x_out, prev_x = self.hs(x)

        if self.reparam_weight:
            x_lvar = self.h_out_lvar(prev_x)
            x_out = torch.stack([x_out, x_lvar])

        # Positional Encoding
        for layer in self.ps:
            x_pos_enc = self.relu(layer(x_pos_enc))
        pos_enc_out = self.p_out(x_pos_enc)

        if self.reparam_pos_enc:
            pos_lvar = self.p_out_lvar(x_pos_enc)
            pos_enc_out = torch.stack([pos_enc_out, pos_lvar])

        return x_out, pos_enc_out

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
    def __init__(self, cfg):
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

        self.gen_in_dim = cfg["set_out_dim"]
        self.gen_hid_dim = cfg["weight_hid_dim"]
        self.gen_layers = cfg["gen_layers"]
        self.norm_lin, self.norm_weights = cfg["norm_lin"], cfg["norm_weights"]
        self.learn_norm = cfg["learn_norm"]
        self.gat_heads = cfg["gat_heads"]
        self.gat_out_dim = cfg["gat_out_dim"]
        weight_bias = cfg["weight_bias"]


        # Weights for final linear classificaion layer
        #print(self.out_sizes[-1][-2])
        self.gat_out_dim = self.gat_out_dim * self.gat_heads
        lin_out_dim = self.gat_out_dim * N_CLASS
        self.w_gen_out = nn.Sequential(
            nn.Linear(self.gen_in_dim, self.gen_hid_dim),
            nn.ReLU(),
            nn.Linear(self.gen_hid_dim, lin_out_dim, bias=(weight_bias != "off"))
        )
        if weight_bias == "zero":
            print("Weight bias init to 0")
            self.w_gen_out[-1].bias.data.fill_(0)

        # learned normalisation
        # Long term average: tensor([0.5807]) tensor([1.1656, 2.0050, 2.2350, 0.1268])
        if self.learn_norm:
            if self.norm_lin:
                self.l_norm = torch.nn.Parameter(torch.tensor([0.75]))


    def forward(self, d2v_embed):
        # d2v_embed.shape = [BS, d2v_out]

        # Weights for linear layer
        lin_weights = self.w_gen_out(d2v_embed)
        if self.norm_lin:
            lin_weights = F.normalize(lin_weights, dim=-1)
            if self.learn_norm:
                lin_weights = lin_weights * self.l_norm

        lin_weights = lin_weights.reshape(-1, N_CLASS, self.gat_out_dim)

        return lin_weights


class GNN2(nn.Module):
    def __init__(self, cfg_dims):
        super().__init__()
        self.GATConv = GATConvFunc()

        gat_heads = cfg_dims["gat_heads"]
        gat_hid_dim = cfg_dims["gat_hid_dim"]
        gat_in_dim = cfg_dims["gat_in_dim"]
        gat_out_dim = cfg_dims["gat_out_dim"]
        gat_layers = cfg_dims["gat_layers"]

        self.gat_layers = nn.ModuleList([])
        self.gat_layers.append(pyg.nn.GATConv(gat_in_dim, gat_hid_dim, heads=gat_heads))
        for _ in range(gat_layers - 2):
            self.gat_layers.append(pyg.nn.GATConv(gat_hid_dim*gat_heads, gat_hid_dim, heads=gat_heads))
        self.gat_layers.append(pyg.nn.GATConv(gat_hid_dim*gat_heads, gat_out_dim, heads=gat_heads))


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

    def forward(self, xs, pos_enc):
        """
        :param xs:              shape = [BS, num_rows, num_xs]
        :param pos_enc:         shape = [BS, num_xs, enc_dim]
        :return output:         shape = [BS, num_rows, num_xs, 2]
        """
        bs, num_rows, num_cols = xs.shape
        #lin_weights = weight_list

        # Flatten xs and append on positional encoding
        pos_enc = pos_enc.unsqueeze(1).repeat(1, num_rows, 1, 1).reshape(bs, num_rows * num_cols, -1)
        xs = xs.reshape(bs, num_rows * num_cols, 1)
        xs = torch.cat([xs, pos_enc], dim=-1)

        # Edges are fully connected graph for each row. Rows are processed independently.
        edge_idx = self.graph_matrix(num_rows, num_cols)

        output = []
        # Forward each batch separately
        for x in xs:
            # Forward each GAT layer
            for layer in self.gat_layers:
                x = layer(x, edge_idx)

            # # Sum GAT node outputs for final predictions.
            # x = x.view(num_rows, num_cols, -1)
            # x = x.sum(-2)
            #
            # # Final linear classification layer
            # x = F.linear(x, final_weight)
            output.append(x)

        output = torch.stack(output)

        output = output.reshape(bs, num_rows, num_cols, -1)
        output = torch.mean(output, dim=-2)
        return output


class ProtoNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gnn_model = GNN2(cfg_dims=cfg)
        self.lin_1 = torch.nn.Linear(16, cfg2.proto_dim)


    # From observations, generates latent embeddings
    def to_embedding(self, xs, pos_enc):
        # Pass through GNN -> average -> final linear
        xs = self.gnn_model(xs, pos_enc)
        embeddings = self.lin_1(xs)
        return embeddings


    # Given meta embeddings and labels, generate prototypes
    def gen_prototypes(self, xs_meta, ys_metas, pos_enc):

        embed_metas = self.to_embedding(xs_meta, pos_enc)

        # Seperate out batches
        self.batch_protos = []
        for embed_meta, ys_meta in zip(embed_metas, ys_metas, strict=True):
            labels = torch.unique(ys_meta)
            embed_protos = [embed_meta[ys_meta == i] for i in labels]

            prototypes = {}
            for embed_proto, label in zip(embed_protos, labels, strict=True):
                prototype = torch.mean(embed_proto, dim=0)
                prototypes[label.item()] = prototype

            self.batch_protos.append(prototypes)


    # Compare targets to prototypes
    def forward(self, xs_targ, pos_enc, max_N_label):
        targ_embeds = self.to_embedding(xs_targ, pos_enc)

        # Loop over batches
        all_probs = []
        for protos, targ_embed in zip(self.batch_protos, targ_embeds, strict=True):
            # Find distance between each prototype by tiling one and interleaving the other.
            # tile prototype: [1, 2, 3 ,1, 2, 3]
            # repeat targets: [1, 1, 2, 2, 3, 3]

            labels, prototypes = protos.keys(), protos.values()
            labels, prototypes = list(labels), list(prototypes)

            N_class = len(protos)
            N_targs = len(targ_embed)

            prototypes = torch.stack(prototypes)
            prototypes = torch.tile(prototypes, (N_targs, 1))

            test = torch.repeat_interleave(targ_embed, N_class, dim=-2)

            # Calc distance and get probs
            distances = -torch.norm(test - prototypes, dim=-1)
            distances = distances.reshape(N_targs, N_class)
            probs = torch.nn.Softmax(dim=-1)(distances)

            # Probs are in order of protos.keys(). Map to true classes.
            true_probs = torch.zeros([cfg2.N_target, max_N_label], dtype=torch.float32)
            true_probs[:, labels] = probs

            all_probs.append(true_probs)

        all_probs = torch.concatenate(all_probs)
        return all_probs


class ModelHolder(nn.Module):
    def __init__(self, cfg_all):
        super().__init__()
        cfg = cfg_all["NN_dims"]

        self.reparam_weight = cfg["reparam_weight"]
        self.reparam_pos_enc = cfg["reparam_pos_enc"]

        self.d2v_model = SetSetModel(cfg=cfg)
        # self.weight_model = WeightGenerator(cfg=cfg)
        self.protonet = ProtoNet(cfg=cfg)

    # Forward Meta set and train
    def forward_meta(self, xs_meta, ys_meta):
        pairs_meta = d2v_pairer(xs_meta, ys_meta.unsqueeze(-1))
        embed_meta, pos_enc = self.d2v_model(pairs_meta)

        self.protonet.gen_prototypes(xs_meta, ys_meta, pos_enc)
        return embed_meta, pos_enc

    def forward_target(self, xs_target, embed_meta, pos_enc, max_N_label):
        # weights_target = self.weight_model(embed_meta)
        #preds_meta = self.gnn_model(xs_target, pos_enc)
        preds = self.protonet.forward(xs_target, pos_enc, max_N_label)

        return preds.view(-1, max_N_label)

    def loss_fn(self, preds, targs):
        cross_entropy = torch.nn.functional.cross_entropy(preds, targs)
        return cross_entropy


def main(all_cfgs, device="cpu", nametag=None):
    save_holder = None

    cfg = all_cfgs["DL_params"]
    bs = cfg["bs"]
    num_rows = cfg["num_rows"]

    cfg = all_cfgs["Settings"]
    ds = cfg["dataset"]
    num_epochs = cfg["num_epochs"]

    val_interval = cfg["val_interval"]
    val_duration = cfg["val_duration"]

    dl = SplitDataloader(
        bs=bs, ds_group=cfg2.ds_group, ds_split="train"
    )
    val_dl = SplitDataloader(
        bs=1, ds_group=cfg2.ds_group, ds_split="test"
    )

    print()
    print("Training data names:", dl)
    print("\nTest data names:", val_dl)


    cfg = all_cfgs["Optim"]
    lr = cfg["lr"]
    eps = cfg["eps"]
    decay = cfg["decay"]

    model = ModelHolder(cfg_all=all_cfgs)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=decay)
    # optim_sched = StepLR(optim, step_size=20, gamma=0.5)

    accs, losses = [], []
    val_accs, val_losses = [], []
    st = time.time()
    for epoch in range(num_epochs):
        duration = time.time() - st
        st = time.time()
        print()
        print(f'{epoch = }, {duration = :.2g}s')

        save_grads = None

        # Train loop
        model.train()
        for xs_meta, ys_meta, xs_target, ys_target, max_N_label in itertools.islice(dl, val_interval):

            # Reshape for dataset2vec
            ys_target = ys_target.reshape(-1)

            # First pass with the meta-set, train d2v and get embedding.
            embed_meta, pos_enc = model.forward_meta(xs_meta, ys_meta)
            # Second pass using previous embedding and train weight encoder
            # During testing, rows of the dataset don't interact.
            ys_pred_targ = model.forward_target(xs_target, embed_meta, pos_enc, max_N_label) # TODO: Handle number of classes.


            loss = model.loss_fn(ys_pred_targ, ys_target)
            loss.backward()

            grads = {n: torch.abs(p.grad) for n, p in model.named_parameters() if p.requires_grad and not p.grad is None}

            optim.step()
            optim.zero_grad()

            # Accuracy recording
            predicted_labels = torch.argmax(ys_pred_targ, dim=1)
            accuracy = (predicted_labels == ys_target).sum().item() / len(ys_target)

            accs.append(accuracy), losses.append(loss.item())

            if save_grads is None:
                save_grads = grads
            else:
                for name, abs_grad in grads.items():
                    save_grads[name] += abs_grad
        print(f"Training accuracy : {np.mean(accs[-val_interval:]) * 100:.2f}%")

        # Validation loop
        model.eval()
        epoch_accs, epoch_losses = [], []
        save_ys_targ, save_pred_labs = [], []
        # for xs_meta, ys_meta, xs_target, ys_target, _ in itertools.islice(val_dl, val_duration):
        #
        #     # Reshape for dataset2vec
        #     ys_target = ys_target.view(-1)
        #
        #     ys_meta = torch.clip(ys_meta, 0, N_CLASS - 1)
        #     ys_target = torch.clip(ys_target, 0, N_CLASS - 1)
        #
        #     # Reshape for dataset2vec
        #     pairs_meta = d2v_pairer(xs_meta, ys_meta)
        #
        #     with torch.no_grad():
        #         embed_meta, pos_enc = model.forward_meta(pairs_meta)
        #
        #         ys_pred_targ = model.forward_target(xs_target, embed_meta, pos_enc).view(-1, N_CLASS)
        #         loss = model.loss_fn(ys_pred_targ, ys_target)#torch.nn.functional.cross_entropy(ys_pred_targ, ys_target.long())
        #
        #     # Accuracy recording
        #     predicted_labels = torch.argmax(ys_pred_targ, dim=1)
        #     accuracy = (predicted_labels == ys_target).sum().item() / len(ys_target)
        #
        #     epoch_accs.append(accuracy), epoch_losses.append(loss.item())
        #     save_ys_targ.append(ys_target)
        #     save_pred_labs.append(predicted_labels)
        #
        # val_losses.append(epoch_losses), val_accs.append(epoch_accs)

        # Average gradients
        for name, abs_grad in save_grads.items():
            save_grads[name] = torch.div(abs_grad, val_interval)
        #
        # print(f'Validation accuracy: {np.mean(val_accs[-1]) * 100:.2f}%')
        # print(model.weight_model.l_norm.data.detach())
        #
        # # Save stats
        # if save_holder is None:
        #     save_holder = SaveHolder(".", nametag=nametag)
        # history = {"accs": accs, "loss": losses, "val_accs": val_accs, "val_loss": val_losses, "epoch_no": epoch}
        # save_holder.save_model(model, optim, epoch=epoch)
        # save_holder.save_history(history)
        # save_holder.save_grads(save_grads)

        # optim_sched.step()


if __name__ == "__main__":


    tag = ""# input("Description: ")

    dev = torch.device("cpu")
    for test_no in range(5):

        print("---------------------------------")
        print("Starting test number", test_no)

        main(all_cfgs=get_config(), device=dev, nametag=tag)

    print("")
    print(tag)
    print("Training Completed")

