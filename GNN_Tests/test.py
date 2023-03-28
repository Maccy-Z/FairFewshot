from torch_geometric.nn import GATConv

model = GATConv(2, 3, heads=5)

for name, param in model.named_parameters():
    print(name, param.shape)

