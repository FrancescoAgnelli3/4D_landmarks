import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import open3d as o3d
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import BaseTransform
import pymeshfix
import plotly.graph_objects as go
from torch_geometric.data.collate import collate


class BP4D(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def processed_dir(self):
        return osp.join(self.root, "processed")
    
    @property
    def processed_dir_slice(self):
        return osp.join(self.root, "processed_slice")

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return [f for f in os.listdir(self.processed_dir) if f.endswith(".pt") and f.startswith("data")]
    
    # def apply_transform_to_dataset(self, transform: BaseTransform, save: bool = True):
    #     """
    #     Apply a transform to each Data object in the dataset. The transform can be a T.Compose object. If save is True, the transformed Data objects will be saved to disk, overwriting the original ones.

    #     Args:
    #         transform (BaseTransform): The transform to apply to each Data object. It can be a T.Compose object.

    #     Returns:
    #         None
    #     """
    #     print(f"Applying transform {transform} to dataset...")
    #     for i in tqdm(range(len(self.processed_file_names))):
    #         try:
    #             data = torch.load(osp.join(self.processed_dir, f"data_{i}.pt"))
    #             data = transform(data)
    #             if save:
    #                 torch.save(data, osp.join(self.processed_dir, f"data_{i}.pt"))
    #         except Exception as e:
    #             print(f"Skipping {i} due to error {e}")


    def apply_transform_to_dataset(self, transform: BaseTransform, save: bool = True):
        """
        Apply a transform to each Data object in the dataset. The transform can be a T.Compose object. If save is True, the transformed Data objects will be saved to disk, overwriting the original ones.

        Args:
            transform (BaseTransform): The transform to apply to each Data object. It can be a T.Compose object.

        Returns:
            None
        """
        print(f"Applying transform {transform} to dataset...")
        for i in tqdm(range(len(self.processed_file_names))):
            try:
                data = torch.load(osp.join(self.processed_dir, f"data_{i}.pt"))
                new_data = []
                for single_data in data:
                    new_data.append(transform(single_data))
                if save:
                    torch.save(new_data, osp.join(self.processed_dir, f"data_{i}.pt"))
            except Exception as e:
                print(f"Skipping {i} due to error {e}")



    def process(self):
        print("Processing data...")
        input("Press ENTER to continue")
        i = 0
        for raw_subject_dir in tqdm(sorted(self.raw_paths)):
            for task in sorted(os.listdir(raw_subject_dir)):
                num_clip = 0
                data_list = []
                for clip in sorted(os.listdir(osp.join(raw_subject_dir, task))):
                    if raw_subject_dir.endswith("F001") and task == "T1":
                        root_path = osp.join(raw_subject_dir, task, clip)
                        landmarks_path = os.path.join(root_path, "mesh_landmarks.txt")

                        landmarks = np.loadtxt(landmarks_path, dtype=np.float32)
                        task = str(task)
                        clip = str(clip)
                        id = str(raw_subject_dir).split("/")[-1]

                        data = Data(
                            subject= id,
                            landmarks=torch.from_numpy(landmarks),
                            task=task,
                            clip=clip,
                            )

                        if self.pre_filter is not None and not self.pre_filter(data):
                            print(f"Skipping {id} due to pre-filtering")
                            continue

                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                            # vertices = data.pos.numpy()
                            # faces = data.face.numpy()
                            landmarks = data.landmarks.numpy()
                        data_list.append(data)
                    else:
                        if num_clip % 10 == 0:
                            # find obj files in raw_subject_dir and use first one

                            root_path = osp.join(raw_subject_dir, task, clip)
                            landmarks_path = os.path.join(root_path, "mesh_landmarks.txt")

                            landmarks = np.loadtxt(landmarks_path, dtype=np.float32)
                            task = str(task)
                            clip = str(clip)
                            id = str(raw_subject_dir).split("/")[-1]

                            data = Data(
                                subject= id,
                                pos=None,
                                face=None,
                                landmarks=torch.from_numpy(landmarks),
                                task=task,
                                clip=clip,
                                )

                            if self.pre_filter is not None and not self.pre_filter(data):
                                print(f"Skipping {id} due to pre-filtering")
                                continue

                            if self.pre_transform is not None:
                                data = self.pre_transform(data)
                                # vertices = data.pos.numpy()
                                # faces = data.face.numpy()
                                landmarks = data.landmarks.numpy()
                            data_list.append(data)
                    num_clip += 1
                if len(data_list) > 0:
                    # data_list, slices = self.collate(data_list)
                    torch.save(data_list, osp.join(self.processed_dir, f"data_{i}.pt"))
                    # torch.save(slices, osp.join(self.processed_dir_slice, f"data_slice_{i}.pt"))
                    i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data