import torch
from torch import nn
import math

torch.manual_seed(0)


class EquivariantLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        # Input shape x = [BS, input_size, in_channel]
        super().__init__()

        self.A = nn.Parameter(torch.ones((in_channel, out_channel)))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.B = torch.zeros((in_channel, out_channel))

        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        self.B = self.B / 10
        self.B = nn.Parameter(self.B)

    def forward(self, x):
        # x.shape = (BS, num_items, size)
        num_items = x.shape[-2]
        ones = torch.ones((num_items, num_items), device=x.device)

        x = x @ self.A - ones @ x @ self.B
        return x


if __name__ == "__main__":
    layer = EquivariantLayer(3, 4)

    test_in = torch.ones((2, 16, 3))
    # test_in[0] = torch.zeros((2, 3))

    out = layer(test_in)

    print(f'{test_in = }')
    print()
    print(f'{out = }')
