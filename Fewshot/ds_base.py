import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import itertools
from AllDataloader2 import SplitDataloader
import os

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
        # xs.shape = [BS, n, i, 1], ys.shape = [BS, n, 1, 1]
        n = xs.shape[1]
        i = xs.shape[2]
        ys_int = ys.clone()
        ys = ys.to(torch.float)

        # Part 1
        f_v_bar = self.f_v_bar(xs)
        f_v_bar = torch.mean(f_v_bar, dim=-3)
        v_bar = self.g_v_bar(f_v_bar)

        f_c_bar = self.f_c_bar(ys)
        f_c_bar = torch.mean(f_c_bar, dim=-3)
        c_bar = self.g_c_bar(f_c_bar)

        # Part 2
        # v_bar.shape = [BS, i, 32]
        v_bar = v_bar.unsqueeze(1)
        v_bar = torch.tile(v_bar, (1, n, 1, 1))
        us1 = torch.cat([v_bar, xs], dim=-1)
        us1 = self.f_u(us1)
        us1 = torch.mean(us1, dim=-2)

        # Part 3
        c_bar = c_bar.unsqueeze(1)
        c_bar = torch.tile(c_bar, (1, n, 1, 1))
        us2 = torch.cat([c_bar, ys], dim=-1)
        us2 = torch.mean(us2, dim=-2)
        us2 = self.f_u(us2)

        us = self.g_u(us1 + us2)  # us.shape = [BS, n, 32]

        # Part 4
        us = us.unsqueeze(2)

        cs = torch.cat([us, ys], dim=-1)
        cs = self.f_c(cs)
        cs = torch.mean(cs, dim=-3)
        cs = self.g_c(cs)   # cs.shape = [BS, 1, 32]

        us = torch.tile(us, (1, 1, i, 1))
        vs = torch.cat([us, xs], dim=-1)
        vs = self.f_v(vs)
        vs = torch.mean(vs, dim=-3)
        vs = self.g_v(vs)   # vs.shape = [BS, i, 32]

        self.vs, self.cs = vs, cs

        # Classification
        vs = self.vs.unsqueeze(1)
        vs = torch.tile(vs, (1, n, 1, 1))
        fz = torch.cat([vs, xs], dim=-1)
        fz = self.f_z(fz)
        fz = torch.mean(fz, dim=-2)
        zs = self.g_z(fz)   # zs.shape = [BS, N, 32]

        self.batch_protos = []
        for z_batch, y_batch in zip(zs, ys_int.squeeze(dim=-1), strict=True):
            protos = torch.empty([2, 32])
            for y_val in [0, 1]:
                mask = (y_batch == y_val).squeeze()
                proto_val = z_batch[mask]
                proto_val = torch.mean(proto_val, dim=0)
                protos[y_val] = proto_val

            self.batch_protos.append(protos)


    def forward_target(self, xs):
        # xs.shape = [BS, n, i, 1]
        n = xs.shape[1]

        vs = self.vs.unsqueeze(1)
        vs = torch.tile(vs, (1, n, 1, 1))
        fz = torch.cat([vs, xs], dim=-1)
        fz = self.f_z(fz)
        fz = torch.mean(fz, dim=-2)
        zs = self.g_z(fz)       # zs.shape = [BS, N, 32]

        batch_probs = []
        for bs_zs, protos in zip(zs, self.batch_protos):
            bs_zs = bs_zs.unsqueeze(-2)
            bs_zs = torch.tile(bs_zs, [1, 2, 1])

            dist = bs_zs - protos
            dist = -torch.norm(dist, dim=-1) ** 2

            batch_probs.append(dist)

        batch_probs = torch.stack(batch_probs)

        return batch_probs


def fit(optim, model, meta_xs, meta_ys, targ_xs, targ_ys):

    model.forward_meta(meta_xs, meta_ys)

    probs = model.forward_target(targ_xs)

    targ_ys = targ_ys.squeeze().flatten()
    probs = probs.view(-1, 2)
    loss = F.cross_entropy(probs, targ_ys.long())

    loss.backward()
    optim.step()
    optim.zero_grad()

    predicted_labels = torch.argmax(probs, dim=1)
    accuracy = (predicted_labels == targ_ys).sum().item() / len(targ_ys)

    return accuracy, loss.item()

def main():
    files = [f for f in os.listdir("./ds_saves") if os.path.isdir(f'{"./ds_saves"}/{f}')]
    existing_saves = sorted([int(f) for f in files if f.isdigit()])  # format: save_{number}
    # print(files, existing_saves)
    save_no = existing_saves[-1] + 1
    save_dir = f'./ds_saves/{save_no}'
    print("Making new save folder at: ")
    print(save_dir)
    os.mkdir(save_dir)


    model = InfModel()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    num_rows, num_targets = 10, 10

    dl = SplitDataloader(
        bs=37, num_rows=3, num_targets=29,
        binarise=True, num_cols=-2, ds_group=tuple([0, 1]), ds_split="train"
    )

    for epoch in range(100):
        print()
        st = time.time()

        # Train loop
        model.train()
        for xs, ys, _ in itertools.islice(dl, 172):
            # Train loop
            # xs.shape = [bs, num_rows+num_targets, num_cols]
            xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
            ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
            # Splicing like this changes the tensor's stride. Fix here:
            xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
            ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
            ys_target = ys_target.view(-1)

            xs_meta, ys_meta = xs_meta.unsqueeze(-1), ys_meta.unsqueeze(-1).unsqueeze(-1)
            xs_target = xs_target.unsqueeze(-1)

            acc, loss = fit(optim, model, xs_meta, ys_meta, xs_target, ys_target)

        print(acc, loss)

        duration = time.time() - st
        print(f'{epoch = }, {duration = :.2g}s')

        torch.save(model, f'{save_dir}/model.pt')

if __name__ == "__main__":
    main()




