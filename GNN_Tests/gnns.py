import torch
from torch_geometric.data import Data
from GAtt_Func import GATConv

torch.manual_seed(123)

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
data = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=data, edge_index=edge_index)

model = GATConv(1, 3)
print(model(data.x, edge_index))


