""" Base class for GMD datasets.

AUTHORS: SWMP project
LICENSE: MIT
"""
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset

CURRENT_DIR = osp.dirname(osp.realpath(__file__))


class GMD_Base(InMemoryDataset):
    r"""GMD base class for creating graph datasets which easily fit into CPU memory.

        Inherit from :class:`torch_geometric.data.InMemoryDataset`.

        Args:
            root (str, optional): Root directory where the dataset should be saved.
                (optional: :obj:`None`)
            name (str, optional): Name of the dataset. (optional: :obj:`None`)
            transform (callable, optional): A function/transform that takes in a
                :class:`~torch_geometric.data.Data` or
                :class:`~torch_geometric.data.HeteroData` object and returns a
                transformed version.
                The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                a :class:`~torch_geometric.data.Data` or
                :class:`~torch_geometric.data.HeteroData` object and returns a
                transformed version.
                The data object will be transformed before being saved to disk.
                (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in a
                :class:`~torch_geometric.data.Data` or
                :class:`~torch_geometric.data.HeteroData` object and returns a
                boolean value, indicating whether the data object should be
                included in the final dataset. (default: :obj:`None`)
            log (bool, optional): Whether to print any console output while
                downloading and processing the dataset. (default: :obj:`True`)

        See Also:
            :class:`torch_geometric.data.InMemoryDataset`
            :class:`torch_geometric.data.Dataset`
        """

    def __init__(self, root=None,
                 name=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 log=True):

        self.root = root
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.log = log

        self._SAVED_PATH = osp.join(osp.abspath(self.root), "processed", self.name)
        self._INPUT_FILE = osp.join(CURRENT_DIR, "..", "data", "matpower", f"{self.name}.m")
        super(GMD_Base, self).__init__(root, transform, pre_transform, pre_filter, log)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        r""" Names of preprocessed files.

        Returns:
            List[str]: List of processed files.
                       First is the one of pt format processed for PyTorch.
                       Second is the list of data objects.
        """
        return [f"{self._SAVED_PATH}/processed.pt",
                f"{self._SAVED_PATH}/data_list.pkl"]

    @property
    def raw_file_names(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def __repr__(self):
        r""" Customize the print of the dataset.

        Returns:
            str: the print of the dataset
        """
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.__class__.__name__}({arg_repr}) {self.name}'
