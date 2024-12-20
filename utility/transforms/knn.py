import torch
from torch_geometric.transforms import KNNGraph
import torch_geometric.nn
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.utils import dense_to_sparse
from torch_geometric.transforms import BaseTransform


class KNNGraphLandmarks(KNNGraph):
    def forward(self, data: Data) -> Data:
        assert "landmarks" in data, "Data object must have landmarks attribute"
        batch = data.batch if "batch" in data else None

        edge_index = torch_geometric.nn.knn_graph(
            data.landmarks,
            self.k,
            batch,
            loop=self.loop,
            flow=self.flow,
            cosine=self.cosine,
            num_workers=self.num_workers,
        )

        if self.force_undirected:
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

        data.landmarks_edge_index = edge_index

        return data

class FullyConnectedGraph(BaseTransform):
    def __init__(self, force_undirected: bool = True):
        self.force_undirected = force_undirected

    def __call__(self, data: Data) -> Data:
        edge_index = dense_to_sparse(torch.ones(data.landmarks.shape[0], data.landmarks.shape[0]))[0]

        if self.force_undirected:
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
        data.edge_index = edge_index
        return data