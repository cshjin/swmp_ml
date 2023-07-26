from torch.nn import Module, ModuleDict, ModuleList
from torch_geometric.nn import HANConv, HGTConv, Linear
from torch_geometric.nn.models import MLP
import numpy as np


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
            metadata=None,
            device="cpu", **kwargs):

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
        self.device = device
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
            type (str, optional): Type of node to predict. Defaults to "bus".

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

    def fit(self, optimizer, loader, epochs=200, **kwargs):
        r""" Training for the model.

        Args:
            optimizer (torch.optimizer): Optimizer for the model.
            loader (pyg.DataLoader): Train DataLoader for the model.
            epoch (int, optional): Number of epochs. Defaults to 200.

        Returns:
            tuple: (list of accuracies, list of rocs, list of losses)
        """
        try:
            from tqdm import tqdm
            import torch.nn.functional as F
            import numpy as np
            from sklearn.metrics import roc_auc_score, accuracy_score
        except ImportError:
            raise ImportError("Package(s) not installed.")

        _setting = kwargs.get("setting", "gic")
        _verbose = kwargs.get("verbose", False)
        _weight_arg = kwargs.get("weight_arg", True)
        _device = kwargs.get("device", "cpu")

        # return a set of lists
        all_acc, all_roc_auc, all_loss = [], [], []

        if _verbose:
            pbar = tqdm(range(epochs), desc="Training", leave=False)
        else:
            pbar = range(epochs)

        self.train()
        for _ in pbar:
            t_loss = 0
            all_true_labels, all_pred_labels = [], []
            for i, data in enumerate(loader):
                data = data.to(_device)
                optimizer.zero_grad()

                # Decide between MLD (reg) or GIC (cls)
                if _setting == "mld":
                    out = self.forward(data)[data.load_bus_mask]
                    loss = F.mse_loss(out, data['y'])
                else:
                    out = self.forward(data, "gmd_bus")

                    # Apply weighted cross entropy loss
                    if _weight_arg and (len(data['y'].bincount()) > 1):
                        weight = len(data['y']) / (2 * data['y'].bincount())
                        loss = F.cross_entropy(out, data['y'], weight=weight)
                    else:
                        loss = F.cross_entropy(out, data['y'])

                    y_true_batch = data['y'].detach().cpu().numpy()
                    y_pred_batch = out.argmax(dim=1).detach().cpu().numpy()
                    all_true_labels.extend(y_true_batch)
                    all_pred_labels.extend(y_pred_batch)

                # update loss
                t_loss += loss.item()
                # update parameters
                loss.backward()
                optimizer.step()

            all_true_labels = np.array(all_true_labels)
            all_pred_labels = np.array(all_pred_labels)

            train_acc = accuracy_score(all_true_labels, all_pred_labels)
            train_roc_auc = roc_auc_score(all_true_labels, all_pred_labels)

            all_acc.append(train_acc)
            all_roc_auc.append(train_roc_auc)
            all_loss.append(t_loss)

            if _verbose:
                pbar.set_postfix({"loss": t_loss, "acc": train_acc, "roc_auc": train_roc_auc})

        return all_acc, all_roc_auc, all_loss

    def evaluate(self, loader, **kwargs):
        r""" Evaluate the model.

        Args:
            loader (pyg.DataLoader): DataLoader to evaluate the model.

        Returns:
            tuple: (accuracy, roc_auc, loss)
        """
        try:
            import torch.nn.functional as F
            from sklearn.metrics import roc_auc_score, accuracy_score
        except ImportError:
            raise ImportError("Package(s) not installed.")

        self.eval()

        _setting = kwargs.get("setting", "gic")
        _weight_arg = kwargs.get("weight_arg", True)

        t_loss = 0
        all_true_labels, all_pred_labels = [], []
        for i, data in enumerate(loader):
            data = data.to(self.device)
            if _setting == "mld":
                out = self.forward(data)[data.load_bus_mask]
                loss = F.mse_loss(out, data['y'])
            else:
                out = self.forward(data, "gmd_bus")
                if _weight_arg and (len(data['y'].bincount()) > 1):
                    weight = len(data['y']) / (2 * data['y'].bincount())
                    loss = F.cross_entropy(out, data['y'], weight=weight)
                else:
                    loss = F.cross_entropy(out, data['y'])

                y_true_batch = data['y'].detach().cpu().numpy()
                y_pred_batch = out.argmax(dim=1).detach().cpu().numpy()
                all_true_labels.extend(y_true_batch)
                all_pred_labels.extend(y_pred_batch)

            t_loss += loss.item()
        all_true_labels = np.array(all_true_labels)
        all_pred_labels = np.array(all_pred_labels)

        acc = accuracy_score(all_true_labels, all_pred_labels)
        roc_auc = roc_auc_score(all_true_labels, all_pred_labels)

        return acc, roc_auc, t_loss / len(loader.dataset)

    def eval_single(self, data):
        r""" Evaluate a signle test data.

        Args:
            data (pyg.HeteroData): Single test data.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        data = data.to(self.device)
        out = self.forward(data, "gmd_bus")
        y_pred = out.argmax(dim=1).detach().cpu().numpy()

        return y_pred
