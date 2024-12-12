from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
from metirc_dacon import *


def train_model(model, optimizer, scheduler, train_loader, val_loader, epoch, device, patience=2):
    model.to(device)
    criterion = nn.BCELoss().to(device)
    best_val_loss = 1000000000000
    best_val_score = 0
    best_model = None
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epochs in range(1, epoch+1):
        model.train()
        train_loss = []
        
        for features, labels in tqdm(iter(train_loader)):

            features = features.float().to(device)
            # print(features.shape)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            output = model(features)
            output = output.squeeze(1)


            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epochs}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}]')
        scheduler.step() #_val_loss
        early_stopping(_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
  
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model

    
    return best_model


def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []

    with torch.no_grad():

        for features, labels in tqdm(iter(val_loader)):

            features = features.float().to(device)
            labels = labels.float().to(device)

            probs = model(features)
            probs = torch.sigmoid(probs)
            probs = probs.squeeze(1)

            loss = criterion(probs, labels)

            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        _val_loss = np.mean(val_loss)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)

        # Calculate AUC score
        auc_score = multiLabel_AUC(all_labels, all_probs)

    return _val_loss, auc_score

class EarlyStopping:

    def __init__(self, patience=1, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self._val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, _val_loss, model):

        score = - _val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(_val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(_val_loss, model)
            self.counter = 0

    def save_checkpoint(self, _val_loss, model):
        
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self._val_loss_min:.6f} --> {_val_loss:.6f}).  Saving model ...')
        self._val_loss_min = _val_loss
        torch.save(model.state_dict(), self.path)