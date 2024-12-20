import torch
from torch_geometric.transforms import KNNGraph
import torch_geometric.nn
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.utils import dense_to_sparse
from torch_geometric.transforms import BaseTransform

class Age2Num(BaseTransform):
    def __init__(self):
        self.age_mapping = {'middle_aged': 0, 'child': 1, 'young': 2, 'senior': 3}

    def __call__(self, data: Data) -> Data:
        age = int(self.age_mapping[data.age])
        data.age_num = age
        return data