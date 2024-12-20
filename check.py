from tqdm import tqdm
from dataset import Headspace
import torch_geometric.transforms as T
from utility import transforms as U

dataset = Headspace(
    root="../.data", 
    pre_transform=T.Compose([
    U.RegisterIntoLandmarksEyes(left_eye_index=36, right_eye_index=45, bottom_index=30), 
    U.CenterInLandmarksCenterOfMass()
    ])
)

# dataset=dataset[:50]
dataset = [d for d in tqdm(dataset) if hasattr(d, 'spline')]

i=0
for d in tqdm(dataset):
    if d.edge_index.dim() == 1:
        print(i)
        print(f"Failed on edge index at index {i}")
    if d.edge_attr.dim() == 1:
        print(i)
        print(f"Failed on edge attr at index {i}")
    if d.fpfh_norm.dim() == 1:
        print(i)
        print(f"Failed on fpfh norm at index {i}")
    if d.pos.dim() == 1:
        print(i)
        print(f"Failed on pos at index {i}")
    if d.landmarks.dim() == 1:
        print(i)
        print(f"Failed on landmarks at index {i}")
    if d.fpfh.dim() == 1:
        print(i)
        print(f"Failed on fpfh at index {i}")
    if d.ohe.dim() == 1:
        print(i)
        print(f"Failed on ohe at index {i}")
    i+=1
    if d.sex.dim() == 2:
        print(i)
        print(f"Failed on sex at index {i}")
    if d.age.dim() == 2:
        print(i)
        print(d.age)
        print(f"Failed on age at index {i}")
    i+=1