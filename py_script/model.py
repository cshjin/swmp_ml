from torch.nn import Module, ModuleDict, ModuleList
from torch_geometric.nn import HANConv, HGTConv, Linear
from torch_geometric.nn.models import MLP


class HGT(Module):
    r""" Heterogeneous Graph Transformer (HGT) model.

    Args:
        hidden_channels (int, optional): Number of hidden channels. Defaults to 64.
        out_channels (int, optional): Number of output channels. Defaults to 1.
        num_conv_layers (int, optional): Number of layers for Conv. Defaults to 2.
        conv_type (str, optional): The type of convolutional layer to use. Defaults to "hgt".
        num_heads (str, optional): Number of heads in HGTConv. Defaults to "2".
        num_mlp_layers (int, optional): Number of layers for MLP. Defaults to 3.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        node_types (tuple, optional): List of string for node types. Defaults to None.
        metadata (list, optional): List of metadata. Defaults to None.
    """

    def __init__(
            self,
            hidden_channels=64,
            out_channels=1,
            num_conv_layers=2,
            conv_type="hgt",
            activation="relu",
            num_heads=2,
            num_mlp_layers=3,
            dropout=0.0,
            node_types=None,
            metadata=None, **kwargs):

        super().__init__()

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.num_conv_layers = num_conv_layers
        self.num_mlp_layers = num_mlp_layers
        self.conv_type = conv_type
        self.activation = activation
        self.dropout = dropout
        self.node_types = node_types
        self.metadata = metadata
        self.group = kwargs.get('group', 'min')

        self.lin_dict = ModuleDict()
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = ModuleList()
        for _ in range(self.num_conv_layers):
            # Chose between HGT or HAN conv
            if self.conv_type == "hgt":
                conv = HGTConv(hidden_channels, hidden_channels, self.metadata, self.num_heads, group=self.group)
            else:
                conv = HANConv(hidden_channels, hidden_channels, self.metadata, self.num_heads, dropout=self.dropout)
            self.convs.append(conv)

        self.flatten = MLP(in_channels=hidden_channels,
                           out_channels=out_channels,
                           hidden_channels=hidden_channels,
                           num_layers=self.num_mlp_layers,
                           dropout=self.dropout, **kwargs)

    def forward(self, data, type="bus"):
        r""" Forward pass of the model.

        Args:
            data (pyg.HeteroData): Input data.

        Returns:
            torch.tensor: Output of the model.
        """

        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        output = x_dict[type]
        return self.flatten(output)

    def reset_parameters(self):
        r""" Reset parameters of the model. """

        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
