import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import time

from dataloader import AdultDataLoader, d2v_pairer
from GAtt_Func import GATConvFunc
from save_holder import SaveHolder
from config import get_config
from AllDataloader import SplitDataloader


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
            print(f'Positional encoding bias init to 0')
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
    def __init__(self, cfg, out_sizes: list):
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

        weight_bias = cfg["weight_bias"]

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
            if self.norm_weights:
                self.w_norm = torch.nn.Parameter(torch.tensor([1., 1.8, 2., 0.25]))
            if self.norm_lin:
                self.l_norm = torch.nn.Parameter(torch.tensor([0.75]))

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
        if self.gen_layers == 1:
            print("Only 1 gen layer, Not using gen_hid_dim")
            module = nn.Sequential(nn.Linear(self.gen_in_dim, tot_params))
        else:
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
                if self.norm_weights:
                    lin, src, dst, bias = F.normalize(lin, dim=0), F.normalize(src, dim=0), F.normalize(dst, dim=0), F.normalize(bias, dim=0)
                    if self.learn_norm:
                        lin, src, dst, bias = lin * self.w_norm[0], src * self.w_norm[1], dst * self.w_norm[2], bias * self.w_norm[3]

                # Reshape each weight matrix
                lin = lin.view(gat_out * gat_heads, gat_in)
                src = src.view(1, gat_heads, gat_out)
                dst = dst.view(1, gat_heads, gat_out)
                bias = bias.view(gat_out * gat_heads)

                batch.append((lin, src, dst, bias))
            layer_weights.append(batch)

        layer_weights = list(zip(*layer_weights))  # [BS, num_layers, tensor[4]]

        # Weights for linear layer
        lin_weights = self.w_gen_out(d2v_embed)
        if self.norm_lin:
            lin_weights = F.normalize(lin_weights, dim=-1)
            if self.learn_norm:
                lin_weights = lin_weights * self.l_norm

        lin_weights = lin_weights.view(-1, self.num_classes, self.gat_out_dim)

        return layer_weights, lin_weights


class GNN(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.GATConv = GATConvFunc()
        self.device = device

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

        # Flatten xs and append on positional encoding
        pos_enc = pos_enc.unsqueeze(1).repeat(1, num_rows, 1, 1).view(bs, num_rows * num_cols, -1)
        xs = xs.view(bs, num_rows * num_cols, 1)
        xs = torch.cat([xs, pos_enc], dim=-1)

        # Edges are fully connected graph for each row. Rows are processed independently.
        edge_idx = self.graph_matrix(num_rows, num_cols).to(self.device)

        output = []
        # Forward each batch separately
        for batch_weights, final_weight, x in zip(gat_weights, lin_weights, xs):

            # Forward each GAT layer
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
    def __init__(self, cfg_all, device="cpu"):
        super().__init__()
        cfg = cfg_all["NN_dims"]

        self.reparam_weight = cfg["reparam_weight"]
        self.reparam_pos_enc = cfg["reparam_pos_enc"]

        gat_heads = cfg["gat_heads"]
        gat_hid_dim = cfg["gat_hid_dim"]
        gat_in_dim = cfg["gat_in_dim"]
        gat_out_dim = cfg["gat_out_dim"]
        gat_layers = cfg["gat_layers"]

        gat_shapes = [(gat_in_dim, gat_hid_dim, gat_heads)] + [(gat_hid_dim, gat_hid_dim, gat_heads) for _ in range(gat_layers - 2)] + [(gat_hid_dim, gat_out_dim, gat_heads)]

        load_d2v = cfg["load_d2v"]
        freeze_model = cfg["freeze_d2v"]
        if load_d2v:
            print()
            print("Loading model. Possibly overriding some config options")
            model_load = cfg["model_load"]
            load = torch.load(f"./dataset2vec/{model_load}")
            state, params = load["state_dict"], load["params"]
            set_h_dim, set_out_dim, d2v_layers = params
            cfg["set_h_dim"] = set_h_dim
            cfg["set_out_dim"] = set_out_dim
            cfg["d2v_layers"] = d2v_layers

            model = SetSetModel(cfg=cfg)

            model.load_state_dict(state, strict=False)

            if freeze_model:
                for fs in model.fs.parameters():
                    fs.requires_grad = False
                for gs in model.gs.parameters():
                    gs.requires_grad = False
                for hs in model.hs.parameters():
                    hs.requires_grad = False

            self.d2v_model = model
        else:
            self.d2v_model = SetSetModel(cfg=cfg)
        self.weight_model = WeightGenerator(cfg=cfg, out_sizes=gat_shapes)
        self.gnn_model = GNN(device=device)

    # Forward Meta set and train
    def forward_meta(self, pairs_meta):
        embed_meta, pos_enc = self.d2v_model(pairs_meta)
        # Reparametrisation trick. Save mean and log_var.
        if self.reparam_weight:
            embed_means, embed_lvar = embed_meta[:, 0], embed_meta[:, 1]
            if self.training:
                std = torch.exp(0.5 * embed_lvar)
                eps = torch.randn_like(std)
                embed_meta = embed_means + eps * std
                self.embed_lvar = embed_lvar
                self.embed_means = embed_means
            else:
                embed_meta = embed_means

        if self.reparam_pos_enc:
            pos_means, pos_lvar = pos_enc[:, 0], pos_enc[:, 1]
            if self.training:
                std = torch.exp(0.5 * pos_lvar)
                eps = torch.randn_like(std)
                pos_enc = pos_means + eps * std
                self.pos_lvar = pos_lvar
                self.pos_means = pos_means
            else:
                pos_enc = pos_means

        return embed_meta, pos_enc

    def forward_target(self, xs_target, embed_meta, pos_enc):
        weights_target = self.weight_model(embed_meta)
        preds_meta = self.gnn_model(xs_target, pos_enc, weights_target)

        return preds_meta

    def loss_fn(self, preds, targs):
        cross_entropy = torch.nn.functional.cross_entropy(preds, targs)

        kl_div: torch.Tensor = 0
        if self.reparam_weight:
            div = 1 + self.embed_lvar - self.embed_means.square() - self.embed_lvar.exp()  # [BS, embed_dim]
            kl_div += torch.mean(-0.5 * torch.sum(div, dim=-1))

        if self.reparam_pos_enc:
            div = 1 + self.pos_lvar - self.pos_means.square() - self.pos_lvar.exp()  # [BS, num_cols, emb_dim]
            kl_div += torch.mean(-0.5 * torch.sum(div, dim=-1))

        return cross_entropy + kl_div


def main(all_cfgs, device="cpu"):
    save_holder = None

    # all_cfgs = get_config()


    cfg = all_cfgs["DL_params"]
    bs = cfg["bs"]
    num_rows = cfg["num_rows"]
    num_targets = cfg["num_targets"]
    ds_group = cfg["ds_group"]
    bal_train = cfg["balance_train"]
    one_v_all = cfg["one_v_all"]

    cfg = all_cfgs["Settings"]
    ds = cfg["dataset"]
    num_epochs = cfg["num_epochs"]

    val_interval = cfg["val_interval"]
    val_duration = cfg["val_duration"]

    if ds == "total":
        dl = SplitDataloader(bs=bs, num_rows=num_rows, num_targets=num_targets, ds_group=ds_group,
                                  balance_train=bal_train, one_v_all=one_v_all, split="train")
        val_dl = SplitDataloader(bs=1, num_rows=num_rows, num_targets=num_targets, ds_group=ds_group,
                                      split="val")
    else:
        raise Exception("Invalid dataset")

    cfg = all_cfgs["Optim"]
    lr = cfg["lr"]
    eps = cfg["eps"]

    model = ModelHolder(cfg_all=all_cfgs, device=device).to(device)
    # model = torch.compile(model)

    optim = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)

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
        for xs, ys in itertools.islice(dl, val_interval):

            xs, ys = xs.to(device), ys.to(device)
            # Train loop
            # xs.shape = [bs, num_rows+num_targets, num_cols]
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
            grads = {n: torch.abs(p.grad) for n, p in model.named_parameters() if p.requires_grad}

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

        # Validation loop
        model.eval()
        epoch_accs, epoch_losses = [], []
        save_ys_targ, save_pred_labs = [], []
        for xs, ys in itertools.islice(val_dl, val_duration):
            xs, ys = xs.to(device), ys.to(device)
            # xs.shape = [bs, num_rows+1, num_cols]

            xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
            ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
            # Splicing like this changes the tensor's stride. Fix here:
            xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
            ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
            ys_target = ys_target.view(-1)

            # Reshape for dataset2vec
            pairs_meta = d2v_pairer(xs_meta, ys_meta)

            with torch.no_grad():
                embed_meta, pos_enc = model.forward_meta(pairs_meta)
                ys_pred_targ = model.forward_target(xs_target, embed_meta, pos_enc).view(-1, 2)
                loss = torch.nn.functional.cross_entropy(ys_pred_targ, ys_target)

            # Accuracy recording
            predicted_labels = torch.argmax(ys_pred_targ, dim=1)
            accuracy = (predicted_labels == ys_target).sum().item() / len(ys_target)

            epoch_accs.append(accuracy), epoch_losses.append(loss.item())
            save_ys_targ.append(ys_target)
            save_pred_labs.append(predicted_labels)

        val_losses.append(epoch_losses), val_accs.append(epoch_accs)

        # Average gradients
        for name, abs_grad in save_grads.items():
            save_grads[name] = torch.div(abs_grad, val_interval)

            # Print some useful stats from validation
            # save_ys_targ = torch.cat(save_ys_targ)
            # save_pred_labs = torch.cat(save_pred_labs)[:20]
            # print("Targets:    ", save_ys_targ[:20])
            # print("Predictions:", save_pred_labs[:20])
        print(f'Validation accuracy: {np.mean(val_accs[-1]) * 100:.2f}%')
        print(model.weight_model.l_norm.data.detach(), model.weight_model.w_norm.data.detach())

        # Save stats
        if save_holder is None:
            save_holder = SaveHolder(".")
        history = {"accs": accs, "loss": losses, "val_accs": val_accs, "val_loss": val_losses, "epoch_no": epoch}
        save_holder.save_model(model, optim)
        save_holder.save_history(history)
        save_holder.save_grads(save_grads)


if __name__ == "__main__":
    from evaluate_real_data import main as eval_main
    import random

    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)

    tag = input("Description: ")

    dev = torch.device("cpu")
    for test_no in range(1):

        print("---------------------------------")
        print("Starting test number", test_no)
        main(all_cfgs=get_config(), device=dev)

    print("")
    print("Training Completed")

    for eval_no in range(1):
        print()
        print("Eval number", eval_no)
        eval_main(save_no=-(eval_no + 1), ds_group=-1)

    print()
    print()

    print(tag)
