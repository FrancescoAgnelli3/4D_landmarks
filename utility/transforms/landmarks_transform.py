import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import scipy.spatial.transform as st
import plotly.graph_objects as go


class CenterInLandmarksCenterOfMass(BaseTransform):
    def forward(self, data: Data) -> Data:
        for store in data.node_stores:
            if hasattr(store, "pos") and hasattr(store, "landmarks"):
                landmarks_center_of_mass = store.landmarks.mean(dim=-2, keepdim=True)
                store.pos -= landmarks_center_of_mass
                store.landmarks -= landmarks_center_of_mass
        return data


class RegisterIntoLandmarksEyes(BaseTransform):
    def __init__(self, left_eye_index, right_eye_index, bottom_index):
        self.left_eye_index = left_eye_index
        self.right_eye_index = right_eye_index
        self.bottom_index = bottom_index

    def forward(self, data: Data) -> Data:
        for store in data.node_stores:
            if hasattr(store, "landmarks"):
                landmarks = store.landmarks
                # fig = go.Figure()

                landmarks = landmarks - landmarks[self.left_eye_index]  # translating landmarks to the left eye, so that left eye will be at (0, 0, 0)
                scale = torch.linalg.norm(landmarks[self.right_eye_index])  # calculating the scale
                landmarks = landmarks / scale

                cos_phi = landmarks[self.right_eye_index][0] / torch.sqrt(landmarks[self.right_eye_index][0] ** 2 + landmarks[self.right_eye_index][1] ** 2)
                phi = torch.rad2deg(torch.acos(cos_phi))
                if landmarks[self.right_eye_index][1] > 0:
                    phi = -phi
                rotation_matrix = torch.as_tensor(st.Rotation.from_euler("z", -phi, degrees=True).as_matrix(), dtype=torch.float32)
                landmarks = torch.matmul(landmarks, rotation_matrix)

                cos_theta = landmarks[self.right_eye_index][0] / torch.sqrt(landmarks[self.right_eye_index][0] ** 2 + landmarks[self.right_eye_index][2] ** 2)
                theta = torch.rad2deg(torch.acos(cos_theta))
                if landmarks[self.right_eye_index][2] < 0:
                    theta = -theta
                rotation_matrix = torch.as_tensor(st.Rotation.from_euler("y", -theta, degrees=True).as_matrix(), dtype=torch.float32)
                landmarks = torch.matmul(landmarks, rotation_matrix)

                cos_psi = landmarks[self.bottom_index][2] / torch.sqrt(landmarks[self.bottom_index][1] ** 2 + landmarks[self.bottom_index][2] ** 2)
                psi = torch.rad2deg(torch.acos(cos_psi))
                if landmarks[self.bottom_index][1] > 0:
                    psi = -psi
                rotation_matrix = torch.as_tensor(st.Rotation.from_euler("x", psi, degrees=True).as_matrix(), dtype=torch.float32)
                landmarks = torch.matmul(landmarks, rotation_matrix)

                # fig.add_trace(
                #     go.Scatter3d(x=landmarks[:, 0], y=landmarks[:, 1], z=landmarks[:, 2], mode="markers+text", marker=dict(size=5, color="green"), text=[str(i) for i in range(landmarks.size(0))])
                # )
                # # fig.add_trace(go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers", marker=dict(size=5, color="yellow")))
                # fig.show()

                store.landmarks = landmarks
                store.scale = scale
        return data


class OneHotEncodingNodes(BaseTransform):
    def forward(self, data: Data) -> Data:
        for store in data.node_stores:
            if hasattr(store, "landmarks"):
                store.ohe = torch.eye(store.landmarks.size(0))
        return data

class LandmarksToPos(BaseTransform):
    r"""data.pos becomes landmarks (functional name: :obj:`sample_points`).

    Args:
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed. (default: :obj:`True`)
    """

    def __init__(
        self,
        remove_faces: bool = True,
    ):
        self.remove_faces = remove_faces

    def forward(self, data: Data) -> Data:
        landmarks = data.landmarks
        data.pos = landmarks

        if self.remove_faces:
            data.face = None

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
