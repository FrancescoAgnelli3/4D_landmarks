import open3d as o3d
import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class Fpfh(BaseTransform):
    def __init__(self, radius=0.1, max_nn=30):
        super(Fpfh, self).__init__()
        self.radius = radius
        self.max_nn = max_nn

    def __call__(self, data: torch.Any) -> torch.Any:
        return self.forward(data)
    
    def forward(self, data: Data) -> Data:
        for store in data.node_stores:
            if hasattr(store, "pos") and hasattr(store, "face") and not hasattr(store, "fpfh") and hasattr(store, "landmarks"):
                landmarks_indexes_in_pos = np.zeros(store.landmarks.shape[0], dtype=np.int64)
                for i in range(store.landmarks.shape[0]):
                    landmarks_indexes_in_pos[i] = np.argmin(np.linalg.norm(store.landmarks - store.landmarks[i], axis=-1))
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(store.pos)
                mesh.triangles = o3d.utility.Vector3iVector(store.face)
                mesh.compute_vertex_normals()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
                pcd.normals = o3d.utility.Vector3dVector(mesh.vertex_normals)
                fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius, max_nn=self.max_nn))
                lmks_fpfh = fpfh.data[:, landmarks_indexes_in_pos]
                store.fpfh = torch.tensor(lmks_fpfh.transpose(1,0), dtype=torch.float32)
        return data
    
class NormalizeFpfh(BaseTransform):
    def __init__(self):
        super(NormalizeFpfh, self).__init__()

    def __call__(self, data: torch.Any) -> torch.Any:
        return self.forward(data)
    
    def forward(self, data: Data) -> Data:
        for store in data.node_stores:
            if hasattr(store, "fpfh"):
                fpfh_norm = torch.zeros(store.landmarks.shape[0],33)
                sum_1 = torch.sum(store.fpfh[:, :11], dim=1, keepdim=True)
                sum_2 = torch.sum(store.fpfh[:, 11:22], dim=1, keepdim=True)
                sum_3 = torch.sum(store.fpfh[:, 22:], dim=1, keepdim=True)
                fpfh_norm[:, :11] = store.fpfh[:, :11] / sum_1
                fpfh_norm[:, 11:22] = store.fpfh[:, 11:22] / sum_2
                fpfh_norm[:, 22:] = store.fpfh[:, 22:] / sum_3
                store.fpfh_norm = fpfh_norm
        return data