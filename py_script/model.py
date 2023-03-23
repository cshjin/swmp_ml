from torch.nn import Module, ModuleDict, Sequential, ModuleList, ReLU, Dropout, BatchNorm1d
from torch_geometric.nn import HANConv, HGTConv, Linear


class HGT(Module):
    """ Heterogeneous Graph Transformer (HGT) model.

    Args:
        hidden_channels (int): Number of hidden channels.
        num_sequential_layers (int): Number of layers in the sequential container
        conv_type (str): The type of convolutional layer to use
        out_channels (int): Number of output channels.
        num_heads (int): Number of heads in HGTConv.
        num_layers (int): Number of layers for HGTConv.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        node_types (list, optional): List of string for node types. Defaults to None.
        metadata (tuple, optional): List of metadata. Defaults to None.
    """

    def __init__(
            self,
            hidden_channels,
            # num_sequential_layers,
            conv_type,
            out_channels,
            num_heads,
            num_conv_layers,
            dropout=0.0,
            node_types=None,
            metadata=None, **kwargs):

        super().__init__()

        self.num_heads = num_heads
        self.num_conv_layers = num_conv_layers
        # self.num_sequential_layers = num_sequential_layers
        self.conv_type = conv_type
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
        
        # functions = [("linear", Linear(hidden_channels, hidden_channels)), ("batch_norm", BatchNorm1d(hidden_channels)), ("relu", ReLU()), ("dropout", Dropout(self.dropout))] * self.num_sequential_layers
        # functions.append(Linear(hidden_channels, out_channels))
        # print(functions)
        # self.flatten = Sequential(OrderedDict(functions))

        self.flatten = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Dropout(self.dropout),
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Dropout(self.dropout),
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Dropout(self.dropout),
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Dropout(self.dropout),
            Linear(hidden_channels, out_channels),
        )

    def forward(self, data):
        """ Forward pass of the model.

        Args:
            data (pyg.HeteroData): Input data.

        Returns:
            torch.tensor: Output of the model.
        """

        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        node_idx = data.node_idx_y

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        batch_node_idx = []
        for s in range(data.num_graphs):
            for idx in node_idx[0]:
                batch_node_idx.append(idx + 19 * s)

        output = x_dict['bus'][batch_node_idx]
        return self.flatten(output)

    def reset_parameters(self):
        """ Reset parameters of the model. """

        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# TODO:
# - Make the following hyper-parameters in the deephyper code:
#   - Number of layers in the sequential container