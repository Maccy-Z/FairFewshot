import torch
import torch.nn as nn
import numpy as np
import random

# device = 'cpu'


class GaussianNoise(nn.Module):
    def __init__(self, std, device):
        super().__init__()
        self.std = std
        self.device = device

    def forward(self, x):
        return x + torch.normal(torch.zeros_like(x), self.std)


class MLP(nn.Module):
    def __init__(self, config, seq_len, num_features, num_outputs=1, device="cpu"):
        super(MLP, self).__init__()
        self.num_outputs = num_outputs
        self.num_features = num_features
        self.seq_len = seq_len
        self.device = device

        self.noise_std = config.get('noise_std', 0.3)
        self.pre_sample_weights = config.get('pre_sample_weights', False)
        self.activation = config.get('activation', torch.nn.Sigmoid)
        self.hidden_dim = config.get('hidden_dim', 3)
        self.num_causes = config.get('num_causes', 2)
        self.num_layers = config.get('num_layers', 2)
        self.is_causal = config.get('is_causal', True)
        self.dropout_prob = config.get('dropout_prob', 0.5)
        self.init_std = config.get('init_std', 1)
        self.pre_sample_causes = config.get('pre_sample_causes', False)
        self.causes_mean = config.get(
            'causes_mean', torch.zeros((self.seq_len, 1, self.num_causes)))
        self.causes_std = config.get(
            'causes_std', torch.ones((self.seq_len, 1, self.num_causes)))
        self.in_clique = config.get('in_clique', True)
        self.y_is_effect = config.get('is_effect', True)
        self.sort_features = config.get('sort_features', True)

        with torch.no_grad():
            def generate_module(layer_idx, out_dim):
                noise = (
                    (GaussianNoise(torch.abs(torch.normal(torch.zeros(
                        size=(1, out_dim), device=device), float(self.noise_std))),
                        device=device) if self.pre_sample_weights
                     else GaussianNoise(float(self.noise_std), device=device))
                )
                return [
                    nn.Sequential(*[
                        self.activation(),
                        nn.Linear(self.hidden_dim, out_dim),
                        noise])
                ]

            if self.is_causal:
                self.hidden_dim = max(
                    self.hidden_dim,
                    self.num_outputs + 2 * self.num_features)
            else:
                self.num_causes = self.num_features

            self.layers = [
                nn.Linear(
                    self.num_causes,
                    self.hidden_dim,
                    device=device)]
            self.layers += [
                module for layer_idx in range(self.num_layers - 1)
                for module in generate_module(layer_idx, self.hidden_dim)]
            if not self.is_causal:
                self.layers += generate_module(-1, self.num_outputs)
            self.layers = nn.Sequential(*self.layers)

            # Initialize Model parameters
            for i, (n, p) in enumerate(self.layers.named_parameters()):
                if len(p.shape) == 2:  # Only apply to weight matrices and not bias
                    dropout_prob = self.dropout_prob if i > 0 else 0.0  # Don't apply dropout in first layer
                    dropout_prob = min(dropout_prob, 0.99)
                    nn.init.normal_(p, std=self.init_std)
                    p *= torch.bernoulli(torch.zeros_like(p) + 1. - dropout_prob)

    def forward(self):
        if self.pre_sample_causes:
            causes = torch.normal(self.causes_mean, self.causes_std.abs()).float()
        else:
            causes = torch.normal(
                0., 1., (self.seq_len, 1, self.num_causes), device=self.device).float()

        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        outputs = outputs[2:]

        if self.is_causal:
            ## Sample num_outputs + num_features nodes from graph if model is causal
            outputs_flat = torch.cat(outputs, -1)

            if self.in_clique:  # select some adjacent nodes
                random_perm = (
                        random.randint(0, outputs_flat.shape[-1]
                                       - self.num_outputs - self.num_features)
                        + torch.randperm(self.num_outputs
                                         + self.num_features, device=self.device))
            else:  # select any nodes from output nodes
                random_perm = torch.randperm(
                    outputs_flat.shape[-1] - 1, device=self.device)

            random_idx_y = (
                list(range(-self.num_outputs, -0)) if self.y_is_effect
                else random_perm[0:self.num_outputs])
            random_idx = random_perm[
                         self.num_outputs:self.num_outputs + self.num_features]

            if self.sort_features:
                random_idx, _ = torch.sort(random_idx)
            y = outputs_flat[:, :, random_idx_y]

            x = outputs_flat[:, :, random_idx]
        else:
            y = outputs[-1][:, :, :]
            x = causes

        # binarize output
        thres = y.median()
        y = (y > thres).float() * 1

        return x, y


class MLPDataLoader:
    def __init__(self, bs, num_rows, num_target, num_cols, config, 
                 device="cpu", split="train", num_models=None, 
                 restore_data=False, save_dir=None):
            self.bs = bs
            self.num_rows = num_rows
            self.num_target = num_target
            self.num_cols = num_cols
            self.num_models = num_models
            
            if restore_data:
                if num_models > 0:
                    models = [torch.load(f'{save_dir}/data_model_{i}.pt') 
                              for i in range(num_models)]
                    self.model = lambda i: models[i]
                else:
                    raise Exception(
                        "Model restoration not available with inifnite number of models")
            
            if num_models == -1:
                self.model = lambda i: MLP(config, num_rows + num_target, self.num_cols)
            else:
                models = [MLP(config, num_rows + num_target, self.num_cols) 
                          for i in range(num_models)]
                self.model = lambda i: models[i]

    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        while True:
            if self.num_models == -1:
                model_idx = [None] * self.bs
                xs, ys = list(zip(*[self.model(i).forward() for i in range(self.bs)]))
            else:
                model_idx = np.random.randint(self.num_models, size=self.bs)
                xs, ys = list(zip(*[self.model(
                    model_idx[i]).forward() for i in range(self.bs)]))

            xs = torch.stack(xs).squeeze()
            ys = torch.stack(ys).squeeze(axis=3)
            yield xs, ys.long(), model_idx
    
    def save_models(self, save_dir):
        if not self.new_models:
            for i in range(self.num_models):
                torch.save(self.model(i), f'{save_dir}/data_model_{i}.pt')
        else:
            raise Exception("Model saving not available if new_models=True")

class MLPRandomDimDataLoader:
    def __init__(self, bs, num_rows, num_target, num_cols_range, config, 
                 device="cpu", split="train"):
            self.bs = bs
            self.num_rows = num_rows
            self.num_target = num_target
            self.num_cols_range = num_cols_range
            self.config = config

    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        while True:
            num_features = np.random.randint(
                self.num_cols_range[0], self.num_cols_range[1])
            model = lambda i: MLP(
                config=self.config, 
                seq_len=self.num_rows + self.num_target, 
                num_features=num_features
            )
            xs, ys = list(zip(*[model(i).forward() for i in range(self.bs)]))

            xs = torch.stack(xs).squeeze()
            ys = torch.stack(ys).squeeze(axis=3)
            yield xs, ys.long(), None
