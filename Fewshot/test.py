self.linear_layer = nn.Sequential(
    nn.Linear(self.in_dim, self.hid_dim),
    nn.ReLU(),
    nn.Linear(self.hid_dim, out_dim)
)