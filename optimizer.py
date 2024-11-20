import numpy as np
from scipy.io import loadmat
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Torchdata
from torch.utils.data import DataLoader
from torchsummary import summary
import os
from time import time
from sklearn.metrics import precision_recall_curve, average_precision_score
import time
from datetime import datetime
import time
from scipy import io
import numpy as np
from spectral import *
import spectral as sp
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
from collections import Counter
import copy
 
from DA_model import DSAD
from dataset import Dataset, calibration, set_datasets
 
 
if __name__=="__main__":
 
    dataset_directory = "C:/study/dataset/"
   
    # train_data
    train_calibrations = []
    train_labels = []
   
    train_folder = f"{dataset_directory}/train"
    for file in os.listdir(train_folder):
        print(f"file = {file}")
        directory = os.path.join(train_folder, file)
        print(f"directory = {directory}")
        if 'LL5XI.JPG' in file:
            continue
    train_label = np.load(f'{directory}/label.npy')
    train_data = np.array(sp.io.envi.open(directory + '/data.hdr', directory + '/data.raw').load())
    train_dr = np.array(sp.io.envi.open(directory + '/DARKREF.hdr', directory + '/DARKREF.raw').load())
    train_wr = np.array(sp.io.envi.open(directory + '/WHITEREF.hdr', directory + '/WHITEREF.raw').load())
    train_calibration = calibration(train_data, train_dr, train_wr)
    train_calibrations.append(train_calibration)
    train_labels.append(train_label)
    train_calibrations=np.array(train_calibrations)
    train_labels=np.array(train_labels)
       
 
    # test_data
    test_folder = f"{dataset_directory}/test"
    test_calibrations = []
    test_labels = []
   
    for file in os.listdir(test_folder):
        print(f"file = {file}")  
        directory = os.path.join(test_folder, file)
        if 'LL5XI.JPG' in file:
            continue
    test_label = np.load(f'{directory}/label.npy')
    test_data = np.array(sp.io.envi.open(directory + '/data.hdr', directory + '/data.raw').load())
    test_dr = np.array(sp.io.envi.open(directory + '/DARKREF.hdr', directory + '/DARKREF.raw').load())
    test_wr = np.array(sp.io.envi.open(directory + '/WHITEREF.hdr', directory + '/WHITEREF.raw').load())
    test_calibration = calibration(test_data, test_dr, test_wr)
    test_calibrations.append(test_calibration)
    test_labels.append(test_label)
    test_calibrations=np.array(test_calibrations)
    test_labels=np.array(test_labels)
   
 
    # validation_data
    validation_folder = f"{dataset_directory}/validation"
    validation_calibrations = []
    validation_labels = []
   
    for file in os.listdir(validation_folder):
        directory = os.path.join(validation_folder, file)
        if 'LL5XI.JPG' in file:
            continue
    validation_label = np.load(f'{directory}/label.npy')
    validation_data = np.array(sp.io.envi.open(directory + '/data.hdr', directory + '/data.raw').load())
    validation_dr = np.array(sp.io.envi.open(directory + '/DARKREF.hdr', directory + '/DARKREF.raw').load())
    validation_wr = np.array(sp.io.envi.open(directory + '/WHITEREF.hdr', directory + '/WHITEREF.raw').load())
    validation_calibration = calibration(validation_data, validation_dr, test_wr)
    validation_calibrations.append(validation_calibration)
    validation_labels.append(validation_label)
    validation_calibrations=np.array(validation_calibrations)
    validation_labels=np.array(validation_labels)
   
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# seed
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
 
# parameter
num_bands = 224
num_layers = 4
rep_dims = 64
patch_size = 1
# normalization = F.normalize(x.reshape(x.size(0), -1), dim=).reshape(-1, num_bands, patch_size, patch_sizea)
noramlization = False
learning_rate = 0.0001
weight_decay = 0
epochs = 30
current_model_save_path = "C:/study/model"
eps = 1e-9
batch_size = 512
current_model_type = "DA"
 
 
# train_dataloader
train_indices = set_datasets(train_calibrations, train_labels, patch_size=patch_size)
train_dataset = Dataset(train_calibrations, train_labels, train_indices, patch_size=patch_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)    
 
# test_dataloader
test_indices = set_datasets(test_calibrations, test_labels, patch_size=patch_size)
test_dataset = Dataset(test_calibrations, test_labels, test_indices, patch_size=patch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
# valid_dataloader
valid_indices = set_datasets(validation_calibrations, validation_labels, patch_size=patch_size)
valid_dataset = Dataset(validation_calibrations, validation_labels, valid_indices, patch_size=patch_size)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
 
# model
dsad_model = DSAD(num_bands, num_layers, rep_dims, patch_size).to(device)
optim = torch.optim.Adadelta(dsad_model.parameters(), learning_rate, weight_decay)
dsad_loss = nn.MSELoss().to(device)
 
# optimizer test
loss_dict = {}
# optimizer_case = ['adadelta','Adam','nadam']
optimizer_case = ['adadelta']
optimizer_dict = {}
optimizer_dict['adadelta'] = torch.optim.Adadelta(dsad_model.parameters(),lr=1.0, rho=0.9, eps=1e-06, weight_decay=0, foreach=None,maximize=False, differentiable=False)
# optimizer_dict['Adam'] = torch.optim.Adam(dsad_model.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
# optimizer_dict['nadam'] = torch.optim.NAdam(dsad_model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004, decoupled_weight_decay=False, foreach=None, capturable=False, differentiable=False)
 
for key in optimizer_case:
    loss_dict[key] = []
 
   
# train_loss 값을 따로 관리하는 대신 전체 train_loss를 저장할 변수를 초기화합니다.
train_loss_total = []
# Train
start_now = datetime.now()
start_now = start_now.today().strftime("%y-%m-%d-%H:%M:%S")  
print(f"{start_now} Train start ")
start_time = time.time()

model_score_max = 0
best_epoch = 1
early_stop_counter = 0

start_now2 = start_now.replace(':','-')  
current_model_save_path = f"C:/study/model/DA_{start_now2}"
temp_save_path = current_model_save_path
os.mkdir(current_model_save_path)

for optimizer_name, optimizer in optimizer_dict.items():
    print(optimizer_name)
    loss = 0
    for epoch in range(1, epochs + 1):
        print(f"{epoch}/{epochs}")
        loss_total = []
        dsad_model.train()
        
        for _, (X, Y, abnormality, _) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            abnormality = abnormality.to(device)
 
            dist = dsad_model(X)
            loss = torch.where(abnormality == 1, (dist + eps)**-1, dist).mean() # abnormality: 1(abnormal), -1(normal)
    
            optim.zero_grad()
            loss.backward()
            optim.step()
    
        loss_total.append(loss.item())
        print(f"Epoch: {epoch}/{epochs}, Loss: {loss.item():.5f}({np.mean(loss_total):.5f})")
 
        # Loss
        train_loss_total.append(np.mean(loss_total))
        loss_dict[optimizer_name].append(np.mean(loss_total))
        print({"train_loss": np.mean(loss_total)})
        print(loss_dict)
 
        # Save
        torch.save(current_model_save_path, os.path.join(current_model_save_path, f"DA_{epoch}.el"))
 
        model_score = loss
        if model_score > model_score_max:
            model_score_max = model_score
            best_epoch = epoch
 
    print(f"Best epoch: {best_epoch}/{epochs}")
    torch.save(current_model_save_path, os.path.join(current_model_save_path, f"DA.el"))
 
    results = []
   
    # Test
    indices = []
    loss_total = []
    preds_cls = []
    preds_softmax = []
    scores = []
    labels_multi = []
    labels_binary = []
    time_list = []
 
    loader = val_loader
    with torch.no_grad():
        total_batch = len(loader)
        dsad_model.eval()
        for batch_idx, (X, Y, abnormality, pos) in enumerate(loader):                    
            X = X.to(device)
            Y = Y.to(device)
            abnormality = abnormality.to(device)
 
            dist = dsad_model(X)
            loss = torch.where(abnormality == 1, (dist + eps)**-1, dist).mean() # abnormality: 1(abnormal), -1(normal)
    
            loss_total.append(loss.item())
    
            scores += dist.to('cpu')
            labels_binary += list(torch.where(abnormality == -1, 2, 3).to('cpu').numpy())
            indices += list(zip(*pos))
    
            loss = torch.where(abnormality == 1, (dist + eps)**-1, dist).mean() # abnormality: 1(abnormal), -1(normal)
        loss_total.append(loss.item())
 
   
    scores += dist.to('cpu')
    labels_binary += list(torch.where(abnormality == -1, 2, 3).to('cpu').numpy())
    indices += list(zip(*pos))
 
    loss = torch.where(abnormality == 1, (dist + eps)**-1, dist).mean() # abnormality: 1(abnormal), -1(normal)
    loss_total.append(loss.item())

# 새로운 optimizer로 넘어갈 때 이전 optimizer의 train_loss 값을 유지합니다.
train_loss_total = []


 
preds_cls = np.array(preds_cls)
preds_softmax = np.array(preds_softmax)
scores = np.array(scores) # distance scores
labels_multi = np.array(labels_multi)
labels_binary = np.array(labels_binary)
 
 # AUPR
precision, recall, thresholds = precision_recall_curve(labels_binary-2, scores)
aupr = average_precision_score(labels_binary-2, scores)
 
f1scores = 2.*(recall*precision)/(recall+precision+1e-9)
best_thr = thresholds[np.argmax(f1scores)]
 
# best threshold ad results
preds_ad = np.where(scores < best_thr, 2, 3) # normal < threshold < abnormal
 
preds = preds_ad
labels = labels_binary
 
c = torch.zeros(dsad_model.rep_dim, device=device)
c_size = 0
dsad_model.eval()
with torch.no_grad():
    for _, (X, _, abnormality, _) in enumerate(train_loader):
        X = X[torch.where(abnormality == -1)].to(device)
        c_size += X.size(0)
        c += torch.sum(torch.flatten(dsad_model.encoder(X.to(device))), dim=0).detach()
c /= c_size
 
c[(abs(c) < 0.1) & (c < 0)] = -0.1
c[(abs(c) < 0.1) & (c > 0)] = 0.1
 
dsad_model.c = nn.Parameter(c)
 
print(f"Model has been saved in {current_model_save_path}")
 
# visualization
markers = {'adadelta' : 'o', 'Adam' : 'x','nadam' : 's'}
plt.figure(figsize = (10,5))
for key in optimizer_case:
    plt.plot(range(1, epochs+1), loss_dict[key], marker = markers[key], markevery = 100, label = key)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
 
