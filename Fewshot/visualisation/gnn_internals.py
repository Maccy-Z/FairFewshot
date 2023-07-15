# Visualising the internal state of the model. Save states in custom model.
import os
import sys
sys.path.append("/mnt/storage_ssd/FairFewshot")

import torch
from Fewshot.AllDataloader import SplitDataloader
from old.dataloader import d2v_pairer
import toml
from Fewshot.main import GATConvFunc, ModelHolder, get_config
import networkx as nx
from matplotlib import pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class GNN2(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.GATConv = GATConvFunc()
        self.device = device
        self.save_alpha = []

    # Generate additional fixed embeddings / graph
    @staticmethod
    def graph_matrix(num_rows, num_xs):
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
        self.weight_list = weight_list

        bs, num_rows, num_cols = xs.shape
        gat_weights, lin_weights = weight_list

        # Flatten xs and append on positional encoding
        pos_enc = pos_enc.unsqueeze(1).repeat(1, num_rows, 1, 1).view(bs, num_rows * num_cols, -1)
        xs = xs.view(bs, num_rows * num_cols, 1)
        xs = torch.cat([xs, pos_enc], dim=-1)

        # Edges are fully connected graph for each row. Rows are processed independently.
        edge_idx = self.graph_matrix(num_rows, num_cols).to(self.device)

        self.save_alpha = []
        output = []
        # Forward each batch separately
        for batch_weights, final_weight, x in zip(gat_weights, lin_weights, xs):

            # Forward each GAT layer
            for layer_weights in batch_weights:
                lin_weight, src_weight, dst_weight, bias_weight = layer_weights

                x = self.GATConv(x, edge_idx, lin_weight, src_weight, dst_weight, bias_weight)
                self.save_alpha.append(self.GATConv.save_alpha)

            # Sum GAT node outputs for final predictions.
            x = x.view(num_rows, num_cols, -1)
            x = x.sum(-2)

            # Final linear classification layer
            x = F.linear(x, final_weight)
            output.append(x)

        output = torch.stack(output)
        return output


class Fewshot:
    def __init__(self, save_dir):
        print(f'Loading model at {save_dir = }')

        state_dict = torch.load(f'{save_dir}/model.pt')
        self.model = ModelHolder(cfg_all=get_config(cfg_file=f'{save_dir}/defaults.toml'))
        self.model.gnn_model = GNN2()
        self.model.load_state_dict(state_dict['model_state_dict'])


    def fit(self, xs_meta, ys_meta):
        pairs_meta = d2v_pairer(xs_meta, ys_meta)

        with torch.no_grad():
            self.embed_meta, self.pos_enc = self.model.forward_meta(pairs_meta)

    def get_acc(self, xs_target, ys_target):
        with torch.no_grad():
            ys_pred_target = self.model.forward_target(xs_target, self.embed_meta, self.pos_enc)

        ys_pred_target_labels = torch.argmax(ys_pred_target.view(-1, 2), dim=1)
        accuracy = (ys_pred_target_labels == ys_target).sum().item() / len(ys_target)

        return accuracy, self.model.gnn_model.save_alpha, self.model.gnn_model.weight_list

    def __repr__(self):
        return "Fewshot"


def get_batch(dl, num_rows):
    batch = next(iter(dl))
    xs, ys, _ = batch

    xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
    ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
    xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
    ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
    ys_target = ys_target.view(-1)

    return xs_meta, xs_target, ys_meta, ys_target


def main(save_no, ds_name="adult"):
    BASEDIR = '.'
    dir_path = f'{BASEDIR}/saves'
    files = [f for f in os.listdir(dir_path) if os.path.isdir(f'{dir_path}/{f}')]
    existing_saves = sorted([int(f[5:]) for f in files if f.startswith("save")])  # format: save_{number}
    save_no = existing_saves[save_no]
    save_dir = f'{BASEDIR}/saves/save_{save_no}'


    cfg = toml.load(os.path.join(save_dir, '../defaults.toml'))["DL_params"]


    num_rows = 50 # cfg["num_rows"]
    num_targets = cfg["num_targets"]
    # ds_group = 2 # cfg["ds_group"]

    model = Fewshot(save_dir)

    val_dl = SplitDataloader(ds_group=ds_name, bs=1, num_rows=num_rows, num_targets=1, binarise=True,
                             num_cols=-3)

    save_alphas = [[], []]
    for j in range(1):
        # Fewshot predictions
        xs_meta, xs_target, ys_meta, ys_target = get_batch(val_dl, num_rows)
        model.fit(xs_meta, ys_meta)
        acc, alpha, weight_list = model.get_acc(xs_target, ys_target)

        for a, save_alpha in enumerate(save_alphas):
            save_alphas[a].append(alpha[a].clone())

    for alpha in save_alphas:
        alpha = torch.stack(alpha)
        num_alphas = np.sqrt(alpha.shape[1])

        a_std, a_mean = torch.std_mean(alpha, dim=0)
        a_mean = a_mean[:, 0] #  torch.mean(alpha, dim=-1)
        a_mean = 20 * (a_mean)  # - torch.mean(a_mean)
        edge_matrix = model.model.gnn_model.GATConv.edge_index.T.numpy()

        G = nx.from_edgelist(edge_matrix)

        # labels = ["Age", "WorkClass", "fnlwgt", "Edu", "Edu-num", "Marital", "Occupation", "Relation",
        #           "Race", "Sex", "Cap-gain", "Cap-loss", "Hr/Wk", "NativeCont"]

        print(num_alphas)
        labels = ["Temp", "Nausea", "Lumbar pain", "Urine", "Micturition", "Urethra"]
        assert len(labels) == num_alphas


        node_labels = {node: labels[node] for node in G.nodes()}
        plt.figure(figsize=(12, 12))

        pos = nx.circular_layout(G)
        print(pos)

        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=edge_matrix, width=a_mean)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=25, font_color="orange", verticalalignment="top")
        #plt.axis("off")
        plt.show()

        exit(4)

    return weight_list




if __name__ == "__main__":
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    path = "./datasets/data"
    files = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    print(files)

    save_weights = {}


    ds_file = ["acute-inflammation"]
    print()
    print(ds_file)
    try:
        w_l = main(save_no=10,  ds_name=ds_file)
        save_weights[ds_file] = w_l
    except RuntimeError as e:
        print(e)
        exit(2)

    # with open("./saves/aaab_1/weights", "wb") as f:
    #     pickle.dump(save_weights, f)
    #
    # print(save_weights)



