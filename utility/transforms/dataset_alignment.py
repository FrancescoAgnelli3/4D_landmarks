from typing import Iterable
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from ..spatial.procrustes import procrustes
import plotly.graph_objects as go
import numpy as np


class Procrustes(BaseTransform):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, data: torch.Any) -> torch.Any:
        return self.forward(data, self.model)

    def forward(self, data: Data, model) -> Data:
        for store in data.node_stores:
            if hasattr(store, "landmarks") and hasattr(store, "scale"):
                landmarks = store.landmarks

                R, s, n = procrustes(model, landmarks)
                landmarks /= n
                landmarks = (torch.matmul(landmarks, torch.as_tensor(R, dtype=torch.float32).T) * s) * n

                store.landmarks = landmarks
                store.scale = store.scale / s
        return data
    

