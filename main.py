from tqdm import tqdm, trange
from dataset import BP4D
import torch_geometric.transforms as T
from utility import transforms as U
from utility.train_eval import train, eval
from torch_geometric.loader import DataLoader
import torch
from torch_geometric.data.collate import collate
import numpy as np
import json
import os
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


torch.manual_seed(0)

# from models.Graph_NODE import TemporalGraphODE
from models.Euler_method import TemporalGraphEuler

path = ".data"
dataset = BP4D(
    root=path,
    )

# dataset.apply_transform_to_dataset(U.RegisterIntoLandmarksEyes(left_eye_index=4, right_eye_index=12, bottom_index=83))

# print("Procrustes...")
# print("Computing mean model...")
# model = torch.zeros(84, 3).to('cpu')
# for data in tqdm(dataset):
#     landmarks = data[0].landmarks.to('cpu')
#     model += landmarks
# model /= len(dataset)

# dataset.apply_transform_to_dataset(U.Procrustes(model))

train_dataset = dataset
test_dataset = []
for i in range(len(dataset)):
    if dataset[i][0].task == "T1" and dataset[i][0].subject == "F001":
        for j in range(len(dataset[i])): 
            if j%10==0:
                test_dataset.append(dataset[i][j])

# train_dataset = train_dataset[:int(len(train_dataset) * 0.2)]

train_loader = DataLoader(train_dataset, batch_size=128)

# model = TemporalGraphODE(3, 3, 10, hidden_dim=8)
model = TemporalGraphEuler(3, 3, 10, 3)
model_name = "Euler_method"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.L1Loss()
scheduler = ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

# train
model.train()
for epoch in trange(101):
    acc_test = []
    epoch_accuracy = 0
    loss = train(model, train_dataset, optimizer, criterion, device)

    model.eval()
    test_loss = eval(model, test_dataset, criterion, device)
    scheduler.step(loss)

    print(f"Epoch {epoch} - Train Loss: {loss:.4f} - Test Loss: {test_loss:.4f}")
