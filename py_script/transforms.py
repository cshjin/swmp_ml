from torch_geometric.transforms import BaseTransform
from typing import List, Union
from torch_geometric.data import Data, HeteroData


class NormalizeColumnFeatures(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
        dim (int): The dimension to normalize the attributes along. (default: 0)
    """

    def __init__(self, attrs: List[str] = ["x"],
                 dim: int = 0):
        self.attrs = attrs
        self.dim = dim

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value = value - value.min()
                value.div_(value.sum(dim=self.dim, keepdim=True).clamp_(min=1.))
                store[key] = value
        return data
