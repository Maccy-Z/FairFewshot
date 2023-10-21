"""Implementation of Iwata et al. 2020 Meta-learning from Tasks with Heterogeneous Attribute Spaces"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import itertools
from AllDataloader import SplitDataloader
import os

N_class = 3


class ff_block(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 32)

    def forward(self, xs):
        xs = self.l1(xs)
        xs = F.relu(xs)
        xs = self.l2(xs)
        xs = F.relu(xs)
        xs = self.l3(xs)
        return xs


class InfModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.f_v_bar = ff_block(1)
        self.g_v_bar = ff_block(32)

        self.f_c_bar = ff_block(1)
        self.g_c_bar = ff_block(32)

        self.f_u = ff_block(33)
        self.g_u = ff_block(32)

        self.f_v = ff_block(33)
        self.g_v = ff_block(32)

        self.f_c = ff_block(33)
        self.g_c = ff_block(32)

        self.f_z = ff_block(33)
        self.g_z = ff_block(32)

    def forward_meta(self, xs, ys):
        n = xs.shape[1]
        i = xs.shape[2]
        ys_int = ys.clone()
        ys = ys.to(torch.float)

        f_v_bar = self.f_v_bar(xs)
        f_v_bar = torch.mean(f_v_bar, dim=-3)
        v_bar = self.g_v_bar(f_v_bar)

        f_c_bar = self.f_c_bar(ys)
        f_c_bar = torch.mean(f_c_bar, dim=-3)
        c_bar = self.g_c_bar(f_c_bar)

        v_bar = v_bar.unsqueeze(1)
        v_bar = torch.tile(v_bar, (1, n, 1, 1))
        us1 = torch.cat([v_bar, xs], dim=-1)
        us1 = self.f_u(us1)
        us1 = torch.mean(us1, dim=-2)
        c_bar = c_bar.unsqueeze(1)
        c_bar = torch.tile(c_bar, (1, n, 1, 1))
        us2 = torch.cat([c_bar, ys], dim=-1)
        us2 = torch.mean(us2, dim=-2)
        us2 = self.f_u(us2)

        us = self.g_u(us1 + us2)

        us = us.unsqueeze(2)

        cs = torch.cat([us, ys], dim=-1)
        cs = self.f_c(cs)
        cs = torch.mean(cs, dim=-3)
        cs = self.g_c(cs)

        us = torch.tile(us, (1, 1, i, 1))
        vs = torch.cat([us, xs], dim=-1)
        vs = self.f_v(vs)
        vs = torch.mean(vs, dim=-3)
        vs = self.g_v(vs)

        self.vs, self.cs = vs, cs

        vs = self.vs.unsqueeze(1)
        vs = torch.tile(vs, (1, n, 1, 1))
        fz = torch.cat([vs, xs], dim=-1)
        fz = self.f_z(fz)
        fz = torch.mean(fz, dim=-2)
        zs = self.g_z(fz)

        self.batch_protos = []
        for z_batch, y_batch in zip(zs, ys_int.squeeze(dim=-1), strict=True):
            protos = torch.zeros([N_class, 32])
            for y_val in range(N_class):
                mask = (y_batch == y_val).squeeze()

                proto_val = z_batch[mask]
                if proto_val.numel() == 0:
                    proto_val = torch.zeros([1, 32])

                proto_val = torch.mean(proto_val, dim=0)
                protos[y_val] = proto_val

            self.batch_protos.append(protos)

    def forward_target(self, xs):
        n = xs.shape[1]

        vs = self.vs.unsqueeze(1)
        vs = torch.tile(vs, (1, n, 1, 1))
        fz = torch.cat([vs, xs], dim=-1)
        fz = self.f_z(fz)
        fz = torch.mean(fz, dim=-2)
        zs = self.g_z(fz)

        batch_probs = []
        for bs_zs, protos in zip(zs, self.batch_protos):
            bs_zs = bs_zs.unsqueeze(-2)
            bs_zs = torch.tile(bs_zs, [1, N_class, 1])

            dist = bs_zs - protos

            dist = -torch.norm(dist, dim=-1) ** 2
            batch_probs.append(dist)

        batch_probs = torch.stack(batch_probs)

        return batch_probs


def fit(optim, model, meta_xs, meta_ys, targ_xs, targ_ys):
    model.forward_meta(meta_xs, meta_ys)

    probs = model.forward_target(targ_xs)

    targ_ys = targ_ys.squeeze().flatten()
    probs = probs.view(-1, N_class)

    loss = F.cross_entropy(probs, targ_ys.long())
    loss.backward()
    optim.step()
    optim.zero_grad()

    predicted_labels = torch.argmax(probs, dim=1)
    accuracy = (predicted_labels == targ_ys).sum().item() / len(targ_ys)

    return accuracy, loss.item()


def main():
    files = [f for f in os.listdir("../iwata") if os.path.isdir(f'../iwata/{f}')]
    existing_saves = sorted([int(f) for f in files if f.isdigit()])  # format: save_{number}
    save_no = existing_saves[-1] + 1
    save_dir = f'../iwata/{save_no}'
    print("Making new save folder at: ")
    print(save_dir)
    os.mkdir(save_dir)

    model = InfModel()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    num_rows, num_targets = 10, 15

    dl = SplitDataloader(
        bs=37, num_rows=num_rows, num_targets=num_targets,
        binarise=True, num_cols=-2, fold_no=(0, -1), ds_split="train"
    )

    for epoch in range(75):

        # Train loop
        model.train()
        for xs, ys, _ in itertools.islice(dl, 100):
            xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
            ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
            xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
            ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
            ys_target = ys_target.view(-1)

            xs_meta, ys_meta = xs_meta.unsqueeze(-1), ys_meta.unsqueeze(-1).unsqueeze(-1)
            xs_target = xs_target.unsqueeze(-1)

            acc, loss = fit(optim, model, xs_meta, ys_meta, xs_target, ys_target)

        print(f'{acc:.3g}')
        print(f'{epoch = }')
        torch.save(model, f'{save_dir}/model.pt')

    with open(f'{save_dir}/finish', "w") as f:
        f.write("finish")


if __name__ == "__main__":
    main()
