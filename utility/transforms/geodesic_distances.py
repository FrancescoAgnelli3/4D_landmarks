import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import potpourri3d as pp3d
import plotly.graph_objects as go
import scipy.interpolate
import torch_geometric.utils

class ComputeGeodesicBetweenLandmarks(BaseTransform):
    def __init__(self, scale=False):
        """
        Compute the geodesic distances between the landmarks of a mesh.

        Args:
            scale (bool, optional): If True, points in the original scale will be used to compute the geodesic distances. Defaults to False.
        """

        super(ComputeGeodesicBetweenLandmarks, self).__init__()
        self.scale = scale

    def compute_geodesic_distances(self, landmarks: np.ndarray, pos: np.ndarray, face: np.ndarray) -> torch.Tensor:
        landmarks_indexes_in_pos = np.zeros(landmarks.shape[0], dtype=np.int64)
        for i in range(landmarks.shape[0]):
            landmarks_indexes_in_pos[i] = np.argmin(np.linalg.norm(pos - landmarks[i], axis=-1))

        solver = pp3d.MeshHeatMethodDistanceSolver(pos, face)
        geodesic_distances = torch.zeros((landmarks.shape[0], landmarks.shape[0]), dtype=torch.float32)
        for i in range(landmarks.shape[0]):
            dist = solver.compute_distance(landmarks_indexes_in_pos[i])
            for j in range(landmarks.shape[0]):
                geodesic_distances[i, j] = dist[landmarks_indexes_in_pos[j]]

        return geodesic_distances

    def compute_geodesic_paths(self, landmarks: np.ndarray, pos: np.ndarray, face: np.ndarray) -> torch.Tensor:
        landmarks_indexes_in_pos = np.zeros(landmarks.shape[0], dtype=np.int64)
        for i in range(landmarks.shape[0]):
            landmarks_indexes_in_pos[i] = np.argmin(np.linalg.norm(pos - landmarks[i], axis=-1))
        try:
            solver = pp3d.EdgeFlipGeodesicSolver(pos, face)
            geodesic_paths = []
            for i in range(landmarks.shape[0]):
                paths = []
                for j in range(landmarks.shape[0]):
                    if i != j:
                        path = solver.find_geodesic_path(landmarks_indexes_in_pos[i], landmarks_indexes_in_pos[j])
                        paths.append(path)
                    else:
                        paths.append([])
                geodesic_paths.append(paths)

            geodesic_paths = np.array(geodesic_paths, dtype=object)
            return geodesic_paths
        except Exception as e:
            print(e)
            return None

    def forward(self, data: Data) -> Data:
        for store in data.node_stores:
            if hasattr(store, "landmarks") and hasattr(store, "pos") and hasattr(store, "face"):
                landmarks = store.landmarks.clone().numpy()
                pos = store.pos.clone().numpy()
                face = store.face.numpy()

                # landmarks_indexes_in_pos = np.zeros(landmarks.shape[0], dtype=np.int64)
                # for i in range(landmarks.shape[0]):
                #     landmarks_indexes_in_pos[i] = np.argmin(np.linalg.norm(pos - landmarks[i], axis=-1))

                # solver = pp3d.MeshHeatMethodDistanceSolver(pos, face)
                # geodesic_distances = torch.zeros((landmarks.shape[0], landmarks.shape[0]), dtype=torch.float32)
                # for i in range(landmarks.shape[0]):
                #     dist = solver.compute_distance(landmarks_indexes_in_pos[i])
                #     for j in range(landmarks.shape[0]):
                #         geodesic_distances[i, j] = dist[landmarks_indexes_in_pos[j]]
                if self.scale is True:
                    assert hasattr(store, "scale"), "The store does not have the scale attribute."

                    landmarks *= store.scale.numpy()
                    pos *= store.scale.numpy()
                    # if not hasattr(store, "geodesic_path_original_scale"):
                    #     store.geodesic_path_original_scale = self.compute_geodesic_paths(landmarks, pos, face)
                    if not hasattr(store, "geodesic_distances_heat_original_scale"):
                        store.geodesic_distances_heat_original_scale = self.compute_geodesic_distances(landmarks, pos, face)
                else:
                    if not hasattr(store, "geodesic_path"):
                        store.geodesic_path = self.compute_geodesic_paths(landmarks, pos, face)
                    if not hasattr(store, "geodesic_distances_heat"):
                        store.geodesic_distances_heat = self.compute_geodesic_distances(landmarks, pos, face)
            else:
                print("The store does not have the required attributes to compute the geodesic distances.")
        return data


class ComputeSplineFromGeodesicPaths(BaseTransform):
    def __init__(self, num_points: int = 50):
        """
        Compute the spline from the geodesic paths.

        Args:
            num_points (int, optional): Number of points to sample from the spline. Defaults to 50.
        """
        super(ComputeSplineFromGeodesicPaths, self).__init__()
        self.num_points = num_points

    def forward(self, data: Data) -> Data:
        assert hasattr(data, "node_stores"), "Data object must have node_stores attribute."
        for store in data.node_stores:
            if hasattr(store, "geodesic_path"):
                geodesic_path = store.geodesic_path
                spline_mat = np.zeros((geodesic_path.shape[0], geodesic_path.shape[1], self.num_points * 3))
                for i in range(geodesic_path.shape[0]):
                    for j in range(geodesic_path.shape[1]):
                        if j > i:
                            path3d = geodesic_path[i, j]
                            if len(path3d) > 0:
                                path3dx = path3d[:, 0]
                                path3dy = path3d[:, 1]
                                path3dz = path3d[:, 2]
                                lenx = len(path3dx)
                                assert lenx == len(path3dy)
                                assert lenx == len(path3dz)

                                x = np.arange(0, lenx)
                                if lenx < 4:
                                    interpolationx = scipy.interpolate.make_interp_spline(x, path3dx, k=lenx - 1)
                                    interpolationy = scipy.interpolate.make_interp_spline(x, path3dy, k=lenx - 1)
                                    interpolationz = scipy.interpolate.make_interp_spline(x, path3dz, k=lenx - 1)
                                else:
                                    interpolationx = scipy.interpolate.make_interp_spline(x, path3dx)
                                    interpolationy = scipy.interpolate.make_interp_spline(x, path3dy)
                                    interpolationz = scipy.interpolate.make_interp_spline(x, path3dz)
                                val = np.linspace(0, lenx - 1, self.num_points)
                                splinex = interpolationx(val)
                                spliney = interpolationy(val)
                                splinez = interpolationz(val)
                                spline_ij = np.zeros((self.num_points * 3))
                                spline_ij[::3] = splinex
                                spline_ij[1::3] = spliney
                                spline_ij[2::3] = splinez
                                spline_ji = np.zeros((self.num_points * 3))
                                spline_ji[::3] = splinex[::-1]
                                spline_ji[1::3] = spliney[::-1]
                                spline_ji[2::3] = splinez[::-1]
                                spline_mat[i, j] = spline_ij
                                spline_mat[j, i] = spline_ji
                spline_mat = torch.from_numpy(spline_mat).float()
                # convert to edge_attr
                store.spline = spline_mat
                store.edge_attr = spline_mat.reshape(-1, 150)
            else:
                print("The store does not have the required attributes to compute the spline.")
        return data