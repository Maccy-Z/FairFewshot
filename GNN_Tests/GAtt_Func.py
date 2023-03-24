from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax, is_sparse

from torch_geometric.nn.inits import glorot, zeros

import torch.nn.functional as tf


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # self.lin_src = Linear(in_channels, heads * out_channels,
        #                       bias=False, weight_initializer='glorot')
        # self.lin_dst = self.lin_src

        # The learnable parameters to compute attention coefficients:
        # self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        # self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, lin_weight, att_src, att_dst, bias,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:

        self.lin_src = tf.linear(x, lin_weight)
        x_src = x_dst = self.lin_src.view(-1, H, C)

        x = (x_src, x_dst)
        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        self.att_src, self.att_dst = att_src, att_dst
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        # We only want to add self-loops for nodes that appear both as
        # source and target nodes:
        num_nodes = x_src.size(0)

        num_nodes = min(num_nodes, x_dst.size(0))

        # Add self-loops onto graph.
        edge_index, edge_attr = remove_self_loops(
            edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value=self.fill_value,
            num_nodes=num_nodes)

        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        out = out.view(-1, self.heads * self.out_channels)

        out = out + bias

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (torch.Tensor or SparseTensor): A :class:`torch.Tensor`,
                a :class:`torch_sparse.SparseTensor` or a
                :class:`torch.sparse.Tensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is a :obj:`torch.Tensor`, its :obj:`dtype`
                should be :obj:`torch.long` and its shape needs to be defined
                as :obj:`[2, num_messages]` where messages from nodes in
                :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is a :class:`torch_sparse.SparseTensor` or
                a :class:`torch.sparse.Tensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :meth:`propagate`.
            size ((int, int), optional): The size :obj:`(N, M)` of the
                assignment matrix in case :obj:`edge_index` is a
                :class:`torch.Tensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :class:`torch_sparse.SparseTensor` or
                a :class:`torch.sparse.Tensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """

        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size,
                                  kwargs)

        msg_kwargs = self.inspector.distribute('message', coll_dict)
        out = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.inspector.distribute('update', coll_dict)
        out = self.update(out, **update_kwargs)

        return out

    def edge_updater(self, edge_index: Adj, **kwargs):
        r"""The initial call to compute or update features for each edge in the
        graph.

        Args:
            edge_index (torch.Tensor or SparseTensor): A :obj:`torch.Tensor`, a
                :class:`torch_sparse.SparseTensor` or a
                :class:`torch.sparse.Tensor` that defines the underlying graph
                connectivity/message passing flow.
                See :meth:`propagate` for more information.
            **kwargs: Any additional data which is needed to compute or update
                features for each edge in the graph.
        """
        size = self._check_input(edge_index, size=None)

        coll_dict = self._collect(self._edge_user_args, edge_index, size,
                                  kwargs)

        edge_kwargs = self.inspector.distribute('edge_update', coll_dict)
        out = self.edge_update(**edge_kwargs)

        return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:

        alpha = alpha_j + alpha_i

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


def main():
    in_dim, out_dim = 1, 3
    heads = 1

    src = torch.rand((1, heads, out_dim))
    dst = torch.rand((1, heads, out_dim))

    linear_weight = torch.rand((out_dim, in_dim))
    bias = torch.rand(out_dim)

    model = GATConv(in_dim, out_dim)

    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    data = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    print(model(data, edge_index, lin_weight=linear_weight, att_src=src, att_dst=dst, bias=bias))


if __name__ == "__main__":
    torch.manual_seed(123)
    main()

    # tensor([[0.4632, 0.5067, 0.4027],
    #         [0.1512, 0.1654, 0.1315],
    #         [-0.3556, -0.3889, -0.3091]], grad_fn= < AddBackward0 >)

