""" Heterogeneous graph neural networks for electric grid. """

import pickle

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn import Module, ModuleDict, ModuleList
from torch_geometric.nn import HANConv, HGTConv, Linear
from torch_geometric.nn.models import MLP
from torch_geometric.nn.resolver import activation_resolver
from tqdm import tqdm


class HeteroGNN(Module):
    r""" Heterogeneous Graph Transformer (HGT) model.

    Args:
        hidden_channels (int, optional):    Number of hidden channels. Defaults to 64.
        out_channels (int, optional):       Number of output channels. Defaults to 1.
        conv_type (str, optional):          The type of convolutional layer to use. Defaults to "hgt".
        act (str, optional):                Activation function. Defaults to "relu".
        num_heads (str, optional):          Number of heads in HGTConv. Defaults to "2".
        num_conv_layers (int, optional):    Number of layers for Conv. Defaults to 2.
        num_mlp_layers (int, optional):     Number of layers for MLP. Defaults to 3.
        dropout (float, optional):          Dropout rate. Defaults to 0.0.
        node_types (tuple, optional):       List of string for node types. Defaults to None.
        metadata (list, optional):          List of metadata. Defaults to None.
    """

    def __init__(
            self,
            hidden_channels=64,
            out_channels=1,
            conv_type="hgt",
            act="relu",
            num_heads=2,
            num_conv_layers=2,
            num_mlp_layers=3,
            dropout=0.0,
            metadata=None,
            device="cpu", **kwargs):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        self.act = activation_resolver(act)
        self.num_heads = num_heads
        self.num_conv_layers = num_conv_layers
        self.num_mlp_layers = num_mlp_layers
        self.dropout = dropout
        # self.node_types = node_types
        self.metadata = metadata
        self.device = device
        self.group = kwargs.get('group', 'min')

        self.node_types = self.metadata[0]
        self.edge_types = self.metadata[1]

        self.lin_dict = ModuleDict()
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(-1, self.hidden_channels)

        # self.lin_dict_edge = ModuleDict()
        # for edge_type in self.edge_types:
        #     self.lin_dict_edge["+".join(edge_type)] = Linear(-1, self.hidden_channels)

        self.convs = ModuleList()
        for _ in range(self.num_conv_layers):
            # Chose between HGT or HAN conv
            if self.conv_type == "hgt":
                conv = HGTConv(
                    self.hidden_channels,
                    self.hidden_channels,
                    self.metadata,
                    self.num_heads,
                    group=self.group)
            else:
                conv = HANConv(
                    self.hidden_channels,
                    self.hidden_channels,
                    self.metadata,
                    self.num_heads,
                    dropout=self.dropout)
            self.convs.append(conv)

        self.flatten = MLP(in_channels=self.hidden_channels,
                           out_channels=out_channels,
                           hidden_channels=self.hidden_channels,
                           num_layers=self.num_mlp_layers,
                           dropout=self.dropout,
                           act=self.act,
                           **kwargs)

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
            x_dict[node_type] = self.act(self.lin_dict[node_type](x))

        # for edge_type, x in edge_index_dict.items():
        #     edge_index_dict[edge_type] = self.act(self.lin_dict_edge["+".join(edge_type)](x))

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # activation for each node type
            for node_type, x in x_dict.items():
                x_dict[node_type] = self.act(x)

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
                    true_y = data['y']
                    # Apply weighted cross entropy loss
                    if _weight_arg and (len(true_y.bincount()) > 1):
                        weight = len(true_y) / (2 * true_y.bincount())
                        loss = F.cross_entropy(out, true_y, weight=weight)
                    else:
                        loss = F.cross_entropy(out, true_y)

                    y_true_batch = true_y.detach().cpu().numpy()
                    y_pred_batch = out.argmax(dim=1).detach().cpu().numpy()
                    # y_pred_batch = (out.argmax(dim=1) * data.gic_blocker_bus_mask).detach().cpu().numpy()
                    all_true_labels.extend(y_true_batch)
                    all_pred_labels.extend(y_pred_batch)

                # update loss
                t_loss += loss.item()
                # update parameters
                loss.backward()
                optimizer.step()

            t_loss = t_loss / len(loader)
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


class HPIGNN(HeteroGNN):
    """ Heterogeneous Physics-Informed Graph Neural Network (HPI-GNN) model.

    Args:
        hidden_channels (int, optional):    Number of hidden channels. Defaults to 64.
        out_channels (int, optional):       Number of output channels. Defaults to 1.
        conv_type (str, optional):          The type of convolutional layer to use. Defaults to "hgt".
        act (str, optional):                Activation function. Defaults to "relu".
        num_heads (str, optional):          Number of heads in HGTConv. Defaults to "2".
        num_conv_layers (int, optional):    Number of layers for Conv. Defaults to 2.
        num_mlp_layers (int, optional):     Number of layers for MLP. Defaults to 3.
        dropout (float, optional):          Dropout rate. Defaults to 0.0.
        node_types (tuple, optional):       List of string for node types. Defaults to None.
        metadata (list, optional):          List of metadata. Defaults to None.
        eta (float, optional):              Weight for the physics loss. Defaults to 1e-1.
    """

    def __init__(self, hidden_channels=64,
                 out_channels=1,
                 conv_type="hgt",
                 act="relu",
                 num_heads=2,
                 num_conv_layers=2,
                 num_mlp_layers=3,
                 dropout=0,
                 metadata=None,
                 device="cpu",
                 eta=1e-1, **kwargs):
        try:
            import julia
            jl = julia.Julia(compiled_modules=False)
            from julia import Main
            self.julia_api = Main
        except ImportError:
            raise ImportError("Julia is not installed. Please install Julia first. 'pip install julia pyjulia'")

        self.eta = eta

        super().__init__(
            hidden_channels,
            out_channels,
            conv_type,
            act,
            num_heads,
            num_conv_layers,
            num_mlp_layers,
            dropout,
            metadata,
            device,
            **kwargs)

        self.julia_api.include("re_eval_gic.jl")

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
            import numpy as np
            import torch.nn.functional as F
            from sklearn.metrics import accuracy_score, roc_auc_score
            from tqdm import tqdm
        except ImportError:
            raise ImportError("Package(s) not installed.")

        _setting = kwargs.get("setting", "gic")
        _verbose = kwargs.get("verbose", False)
        _weight_arg = kwargs.get("weight_arg", True)
        _device = kwargs.get("device", "cpu")

        # return a set of lists
        all_acc, all_roc_auc, all_loss = [], [], []
        all_ce, all_pi = [], []
        if _verbose:
            pbar = tqdm(range(epochs), desc="Training", leave=False)
        else:
            pbar = range(epochs)

        self.train()
        for epoch in pbar:
            t_loss = 0
            ce_loss = 0
            pi_loss = 0
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

                    if epoch % 10 == 0:
                        # pi_loss_data = self.eval_loss(y_true_batch, y_pred_batch)
                        # loss_data = ce_loss_data + self.eta * self.eval_loss(y_true_batch, y_pred_batch)
                        loss += self.eta / (epoch + 1)**0.5 * self.eval_loss(y_true_batch, y_pred_batch)
                        pi_loss_data = self.eval_loss(y_true_batch, y_pred_batch) * self.eta / (epoch + 1)**0.5
                    ce_loss_data = F.cross_entropy(out, data['y'])
                # update loss
                t_loss += loss.item()
                ce_loss += ce_loss_data.item()
                pi_loss += pi_loss_data

                # update parameters
                loss.backward(retain_graph=True)
                optimizer.step()

            # average over batches
            t_loss /= len(loader)
            ce_loss /= len(loader)
            pi_loss /= len(loader)

            all_true_labels = np.array(all_true_labels)
            all_pred_labels = np.array(all_pred_labels)

            train_acc = accuracy_score(all_true_labels, all_pred_labels)
            train_roc_auc = roc_auc_score(all_true_labels, all_pred_labels)

            all_acc.append(train_acc)
            all_roc_auc.append(train_roc_auc)
            all_loss.append(t_loss)
            all_ce.append(ce_loss)
            all_pi.append(pi_loss)

            if _verbose:
                pbar.set_postfix({"loss": t_loss,
                                  "ce_loss": ce_loss,
                                  "pi_loss": pi_loss,
                                  "acc": train_acc,
                                  "roc_auc": train_roc_auc})
            pickle.dump({"all_loss": all_loss,
                         "all_acc": all_acc,
                         "all_roc_auc": all_roc_auc,
                         "all_ce": all_ce,
                         "all_pi": all_pi},
                        open("train_log.pkl", "wb"))
        return all_acc, all_roc_auc, all_loss

    def eval_loss(self, true_y, pred_y, loss_func="diff", **kwargs):
        """ Loss of the evaluation function.

        Args:
            true_y (np.array): True labels.
            pred_y (np.array): Predicted labels.
            loss_func (str, optional): Loss function. Defaults to "diff".

        Returns:
            float: Loss of the evaluation function.
        """
        _verbose = kwargs.get("verbose", False)

        gic_str = """\n%% gmd_blocker data\n%column_names% gmd_bus status\nmpc.gmd_blocker = {\n"""
        for idx, v in enumerate(true_y):
            gic_str += f"\t{idx+1}\t{v.item()}\n"
        gic_str += "};"

        # write to file
        with open("epri21_train_true.m", "w") as f1:
            with open("./data/matpower/epri21.m", "r") as f2:
                f1.write(f2.read() + gic_str)

        gic_str = """\n%% gmd_blocker data\n%column_names% gmd_bus status\nmpc.gmd_blocker = {\n"""
        for idx, v in enumerate(pred_y):
            gic_str += f"\t{idx+1}\t{v.item()}\n"
        gic_str += "};"

        with open("epri21_train_pred.m", "w") as f1:
            with open("./data/matpower/epri21.m", "r") as f2:
                f1.write(f2.read() + gic_str)

        true_res = self.julia_api.re_eval("epri21_train_true.m")
        pred_res = self.julia_api.re_eval("epri21_train_pred.m")

        feasible = (pred_res[0] == 1)
        if not feasible:
            if _verbose:
                print("ATTENTION: infeasible solution")
            return true_res[1]

        if loss_func == "diff":
            loss = pred_res[1] - true_res[1]
        elif loss_func == "square":
            loss = (pred_res[1] - true_res[1])**2
        elif loss_func == "abs":
            loss = abs(pred_res[1] - true_res[1])

        if _verbose:
            if pred_res[1] < true_res[1]:
                print("ATTENTION: pred_res < true_res", pred_res[1], true_res[1])
        return loss
