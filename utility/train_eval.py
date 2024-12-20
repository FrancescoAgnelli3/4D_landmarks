import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
import tqdm
import random

def train(model, tr_loader, optimizer, criterion, device):
    model.train()

    prev_h = None
    total_loss = 0
    for batch in tr_loader:
        # Reset gradients from previous step
        model.zero_grad()
        y_pred, y_true = [], []
        loss = 0
        land_pred = []
        for i in range(len(batch)-1):
            snapshot = batch[i].to(device)       
            
            # Perform a forward pass
            y, h, x, snap_land_pred = model.forward(snapshot, prev_h)
            prev_h = h

            y_pred.append(y.cpu())
            target = batch[i+1].landmarks.cpu()
            y_true.append(target)
            land_pred.append(torch.cat(snap_land_pred).detach().cpu())
            L = model.emb[0].weight
            L = L.permute(1,0)*L
            M = model.readout[0].weight
            M = M.permute(1,0)*M
            I = torch.eye(L.shape[0]).to(device)
            norm = torch.linalg.norm(L-I) + torch.linalg.norm(M-I)
            loss += criterion(y, target.to(device))
            loss += criterion(x, snapshot.landmarks)
            loss += norm

        # Perform a backward pass to calculate the gradients
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        land_pred = torch.cat(land_pred)
        loss.backward()
        total_loss += loss.item()

        # Update parameters
        optimizer.step()

        # if you don't detatch previous state you will get backprop error
        if prev_h is not None:
            prev_h = prev_h.detach()
        
    return total_loss/len(tr_loader)


def eval(model, loader, criterion, device):
    total_loss = 0
    with torch.no_grad():
        prev_h = None
        model.zero_grad()
        loss = 0

        land_pred = []
        for i in range(len(loader)-1):
            snapshot = loader[i].to(device)       
            
            # Perform a forward pass
            y, h, x, snap_land_pred = model.forward(snapshot, prev_h)
            prev_h = h

            target = loader[i+1].landmarks.cpu()
            land_pred.append(torch.cat(snap_land_pred).detach().cpu())
            loss += criterion(y, target.to(device))
            loss += criterion(x, snapshot.landmarks)
        land_pred = torch.cat(land_pred)
        total_loss += loss.item()

    torch.save(land_pred, f"/home/agnelli/projects/4D_landmarks/results/pred_land.npy")

    return total_loss
