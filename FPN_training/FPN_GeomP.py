#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import random
import torch
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# sys.setrecursionlimit(10**9)
# check the dependences version
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("python  version: ",sys.version)
print("pytorch version: ",torch.__version__)
print("numpy   version: ",np.__version__)
print("device         : ",device)

# data read and transform to input matrix 
# dont shuffle in get_input_matrixes() and get_y_QP()!!!!!!!!!
def get_input_matrixes(inputPath,cu_w=64,cu_h=64,dtype=float):
    concatenated_df=pd.read_csv(inputPath,header=None)
    samples = concatenated_df.values[:,:]
    input_matrixes = []
    for sample in samples:
        # note the form of pytorch tensor is [N, C, H, W]
        matrix = np.zeros((cu_h,cu_w),dtype=dtype)
        for i in range(cu_h):
            for j in range(cu_w):
                matrix[i,j] = sample[i*cu_w+j]
        input_matrixes.append(matrix)
    return np.array(input_matrixes)

def get_y_QP(inputPath, y_dim=480, qp_w=64, qp_h=64):
    concatenated_df=pd.read_csv(inputPath,header=None)
    samples = concatenated_df.values[:,:]
    y_vectors = []
    qp_matrix = []
    for sample in samples:
        y_temp1 = np.zeros([16,15]) ## ver edge
        y_temp2 = np.zeros([15,16]) ## hor edge
        y_temp3 = np.zeros([480])
        for j in range(15):
            for i in range(16):
                y_temp2[j,i] = sample[240+j*16+i]
                y_temp1[i,j] = sample[j*16+i]
        
        count = 0
        for i in range(16):
            for j in range(16):
                if i < 15:
                    y_temp3[count] = y_temp2[i,j]
                    count += 1
                    if j<15:
                        y_temp3[count] = y_temp1[i,j]
                        count +=1
                elif (i==15) & (j<15):
                    y_temp3[count] = y_temp1[i,j]
                    count += 1

        y_vectors.append(sample[0:y_dim])
        qp_matrix.append(np.ones([qp_h,qp_w])*sample[y_dim:y_dim+1])
    return np.array(qp_matrix),np.array(y_vectors)
        
def data_clean(x, y):
    index = [i for i in range(len(y))]
    for i in range(10):
        random.shuffle(index)
    x = x[index]
    y = y[index]

    counter = np.zeros([480])
    y_calcu = torch.sum(y,axis=1)
    for i in range(y_calcu.shape[0]):
        counter[int(y_calcu[i])] += 1
    counter0 = counter1 = counter2 = counter3 = min([counter[0],np.sum(counter[1:17]),np.sum(counter[17:33]),np.sum(counter[33:481])])
    all_count        = int(4 * counter0)
    real_counter     = 0
    after_y          = torch.zeros([all_count,y.shape[1]],dtype=torch.float32)
    after_x          = torch.zeros([all_count,x.shape[1],x.shape[2],x.shape[3]],dtype=torch.float32)
    for i in range(y.shape[0]):
        if real_counter == all_count:
            break
        now_sum = torch.sum(y[i]).item()
        if (now_sum == 0) & (counter0>0):
            after_y[real_counter] = y[i]
            after_x[real_counter] = x[i]
            counter0             -= 1
            real_counter         += 1
        elif (now_sum>=1) & (now_sum<17) & (counter1>0):
            after_y[real_counter] = y[i]
            after_x[real_counter] = x[i]
            counter1             -= 1
            real_counter         += 1
        elif (now_sum>=17) & (now_sum<33) & (counter2>0):
            after_y[real_counter] = y[i]
            after_x[real_counter] = x[i]
            counter2             -= 1
            real_counter         += 1
        elif (now_sum>=33) & (now_sum<481) & (counter3>0):
            after_y[real_counter] = y[i]
            after_x[real_counter] = x[i]
            counter3             -= 1
            real_counter         += 1

    return after_x, after_y

def build_dataset(toriPath, occuPath, y_QPPath):
    toriData = get_input_matrixes(toriPath,dtype=float)
    occuData = get_input_matrixes(occuPath,dtype=int)
    qp, y    = get_y_QP(y_QPPath)
    dataset  = list(zip(toriData,occuData,qp,y))
    for i in range(10):
        np.random.shuffle(dataset)
    dataset_train, dataset_test, _, _ = train_test_split(dataset, y, test_size=0.05, shuffle=False)
    ## Note: Due to the data clean, the actual number of training sets will be greatly reduced than the number of 95% data sets. Here, the test set is selected as 5% of the data set to increase the number of training set samples and ensure the actual number of samples for training: the number of test samples is about 5:1.

    toriData_train, occuData_train, qp_train, y_train = zip(*dataset_train)
    toriData_train = torch.tensor(np.array(toriData_train), dtype=torch.float32)
    occuData_train = torch.tensor(np.array(occuData_train), dtype=torch.float32)
    qp_train       = torch.tensor(np.array(qp_train), dtype=torch.float32)
    y_train        = torch.tensor(np.array(y_train), dtype=torch.float32)
    inputs_train   = torch.stack([toriData_train,occuData_train,qp_train],axis=3)
    
    toriData_test, occuData_test, qp_test, y_test = zip(*dataset_test)
    toriData_test = torch.tensor(np.array(toriData_test), dtype=torch.float32)
    occuData_test = torch.tensor(np.array(occuData_test), dtype=torch.float32)
    qp_test       = torch.tensor(np.array(qp_test), dtype=torch.float32)
    y_test        = torch.tensor(np.array(y_test), dtype=torch.float32)
    inputs_test   = torch.stack([toriData_test,occuData_test,qp_test],axis=3)

    return inputs_train.permute(0,3,1,2), y_train, inputs_test.permute(0,3,1,2), y_test

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from functools import partial
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class BasicBlock(nn.Module):
    # def __init__(self, input_channel=2, output_channel=2, stride=1, training=False):
    def __init__(self, input_channel=3, output_channel=3, stride=1, training=False, kernel_size=3):
        super().__init__()
        self.conv_middle_channel=64

        self.conv1 = nn.Conv2d(input_channel,self.conv_middle_channel,kernel_size=kernel_size,stride=stride,bias=False)
        self.bn1   = nn.BatchNorm2d(self.conv_middle_channel)
        self.act1  = nn.LeakyReLU(negative_slope=5e-2)
        
        self.conv2 = nn.Conv2d(self.conv_middle_channel,output_channel,kernel_size=kernel_size,stride=1,bias=False)
        self.bn2   = nn.BatchNorm2d(output_channel)
        self.act2  = nn.LeakyReLU(negative_slope=5e-2)
        
    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = out+inputs[:,:,2:out.shape[2]+2,2:out.shape[3]+2]
        out = self.act2(out)
            
        return out
    
    
class FPN(nn.Module):
    def __init__(self):
        super(FPN,self).__init__()
        self.block_c2=nn.Sequential(
            BasicBlock(),
            BasicBlock(),
            BasicBlock(),
            BasicBlock(),
            BasicBlock(),
            BasicBlock(),
            BasicBlock(),
            BasicBlock()
        )
        self.block_c3=nn.Sequential(
            BasicBlock(),
            BasicBlock(),
            BasicBlock(),
            BasicBlock()
        )
        self.block_c4=nn.Sequential(
            BasicBlock(),
            BasicBlock()
        )
        self.block_c5=nn.Sequential(
            BasicBlock()
        )

        self.conv_p5              = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(negative_slope=5e-2)
        )
        self.conv_p4              = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),

            nn.Conv2d(64, 3, kernel_size=3),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(negative_slope=5e-2)
        )
        self.conv_p3              = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 3, kernel_size=3),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(negative_slope=5e-2)
        )
        self.conv_p2              = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 3, kernel_size=3),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(negative_slope=5e-2)
        )        

        self.output_end           = nn.Sequential(
            nn.Conv2d(15, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2),
            
            nn.Conv2d(64, 30, 5),
            nn.BatchNorm2d(30),
            nn.LeakyReLU(negative_slope=5e-2)
            
        )

    def forward(self, inputs):             ## [N, 3, 64, 64]
        ## bottom to top    
        c2 = self.block_c2(inputs)         ## [N, 3, 32, 32]
        c3 = self.block_c3(c2)             ## [N, 3, 16, 16]
        c4 = self.block_c4(c3)             ## [N, 3,  8,  8]
        c5 = self.block_c5(c4)             ## [N, 3,  4,  4]
        
        ## top to bottom
        p5 = self.conv_p5(c5)                                                                             ## [N, 3,  2,  2]
        p4 = self.conv_p4(torch.cat([F.interpolate(p5,scale_factor=4.0,mode='bilinear'), c4],dim=1))      ## [N, 3,  4,  4]
        p3 = self.conv_p3(torch.cat([F.interpolate(p4,scale_factor=4.0,mode='bilinear'), c3],dim=1))      ## [N, 3,  8,  8]
        p2 = self.conv_p2(torch.cat([F.interpolate(p3,scale_factor=4.0,mode='bilinear'), c2],dim=1))      ## [N, 3, 16, 16]

        ## reduce channel and dimension normalization
        h2 = F.interpolate(p2,scale_factor=4.0,mode='bilinear')
        h3 = F.interpolate(p3,scale_factor=8.0,mode='bilinear')
        h4 = F.interpolate(p4,scale_factor=16.0,mode='bilinear')
        h5 = F.interpolate(p5,scale_factor=32.0,mode='bilinear')

        ## output prediction
        x = torch.cat([inputs, h2, h3, h4, h5],dim=1)
        x = self.output_end(x)       ## [32, 30, 4, 4]

        x = torch.flatten(x,start_dim=1,end_dim=3)

        return x
    
class CFPN(nn.Module):
    def __init__(self):
        super(CFPN,self).__init__()
        self.FPN = FPN()
        self.linear1 = nn.Linear(in_features=480, out_features=480)
        self.linear2 = nn.Linear(in_features=480, out_features=480)
        self.act     = nn.LeakyReLU(negative_slope=5e-2)
        
    # def forward(self, inputs):
    def forward(self, inputs):
        x = self.FPN(inputs)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)

        return x
    
class CFPN_Loss(nn.Module):
    def __init__(self, y, reduction='none', version='stand'):
        super().__init__()
        self.reduction  = reduction
        self.version    = version
        self.gamma      = 2
        self.pos_weight = (y.size(0)/(y.sum(dim=0)+1e-5)*0.8).to(device)
        self.neg_weight = (y.size(0)/(y.size(0)-y.sum(dim=0)+1e-5)*1).to(device)
                
    def forward(self, input, target):
        loss=torch.zeros([1,480],dtype=torch.float).to(device)
        if self.version=='stand':
            for i in range(target.size(0)):
                # loss += -(target[i] * torch.log(input[i]+1e-5) + (1 - target[i]) * torch.log(1 - input[i]+1e-5))
                loss += -(self.pos_weight*target[i] * torch.log(input[i]+1e-5) + self.neg_weight*(1 - target[i]) * torch.log(1 - input[i]+1e-5))
        elif self.version=='Focal_Loss':
            for i in range(target.size(0)):
                loss += -((1-target[i])*torch.pow(input[i],self.gamma) * torch.log(1-input[i]+1e-5) + target[i]*torch.pow(1 - input[i],self.gamma) * torch.log(input[i]+1e-5))
        elif self.version=='ablationLoss':
            for i in range(target.size(0)):
                loss += -(target[i] * torch.log(input[i]+1e-5) + (1 - target[i]) * torch.log(1 - input[i]+1e-5))
                        
        if self.reduction=='mean':
            return (loss/target.size(0)).mean()
        else:
            return loss/target.size(0)


#data process
print("\nGeomP training begin: ")
save_file       = "/mnt/disk0/qsc/nnTrainingResult/AI/logs_FPN_GeomP"
writer          = SummaryWriter(save_file)
# toriPath = '/mnt/disk0/qsc/nnDataset/basketball_player_simpleTest/GeomP_Resi.csv'
# occuPath = '/mnt/disk0/qsc/nnDataset/basketball_player_simpleTest/GeomP_Occu.csv'
# y_QPPath = '/mnt/disk0/qsc/nnDataset/basketball_player_simpleTest/GeomP_y_QP.csv'
toriPath = '/mnt/disk0/qsc/nnDataset/basketball_player32/GeomP_Resi.csv'
occuPath = '/mnt/disk0/qsc/nnDataset/basketball_player32/GeomP_Occu.csv'
y_QPPath = '/mnt/disk0/qsc/nnDataset/basketball_player32/GeomP_y_QP.csv'
inputs_train, y_train, inputs_test, y_test = build_dataset(toriPath, occuPath, y_QPPath)

# parameter setting
model         = CFPN().to(device)
optimizer     = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler     = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
max_epoch     = 500
batch_size    = 32

# begin training and testing
last_1recall = 0
last_1precis = 0
early_stop   = 0
threshold    = 0.5
loss_fn      = nn.BCELoss(reduction='none')
for epoch in range(max_epoch):
    clean_inputs_train, clean_y_train = data_clean(inputs_train,y_train)
    clean_y_train.to(device)
    iteration     = lambda : int(clean_y_train.size(0)/batch_size) if int(clean_y_train.size(0)/batch_size)==clean_y_train.size(0)/batch_size else int(clean_y_train.size(0)/batch_size)+1
    print("-------Epoch {}-------".format(epoch))

    # train the data
    model.train()
    train_sum_loss = 0
    train_1recall = 0
    train_0recall = 0
    train_1precis = 0
    train_0precis = 0
    train_acc     = 0
    train_recall  = 0
    for i in range(iteration()):
        range_min = int(i*batch_size)
        range_max = int(i*batch_size)+batch_size
        if clean_y_train.size(0)<range_max:
            range_max = clean_y_train.size(0)
        indices = torch.arange(range_min,range_max).int()
        
        inputs  = torch.index_select(clean_inputs_train, 0, indices).to(device)
        y       = torch.index_select(clean_y_train, 0, indices).to(device)
        outputs = model(inputs).to(device)
        train_TP= torch.where(torch.gt(outputs,threshold),torch.where(torch.gt(outputs,threshold)==y,1,0),0).sum()
        train_FP= torch.where(torch.gt(outputs,threshold),torch.where(torch.gt(outputs,threshold)!=y,1,0),0).sum()
        train_TN= torch.where(torch.le(outputs,threshold),torch.where(torch.gt(outputs,threshold)==y,1,0),0).sum()
        train_FN= torch.where(torch.le(outputs,threshold),torch.where(torch.gt(outputs,threshold)!=y,1,0),0).sum()
        train_1recall += (train_TP+1e-5)/(train_TP+train_FN+1e-5)
        train_0recall += (train_TN+1e-5)/(train_TN+train_FP+1e-5)
        train_1precis += (train_TP+1e-5)/(train_TP+train_FP+1e-5)
        train_0precis += (train_TN+1e-5)/(train_TN+train_FN+1e-5)
        train_acc     += (train_TP+train_TN+1e-5)/(train_TP+train_FP+train_TN+train_FN+1e-5)
        
        train_loss = loss_fn(outputs, y)
        train_sum_loss += torch.sum(train_loss).item()
        
        optimizer.zero_grad()
        train_loss.backward(torch.ones(train_loss.size(0),480).to(device))
        optimizer.step()

    writer.add_scalar(f"train label1 precision", round((train_1precis / iteration()).item(),5), epoch)
    writer.add_scalar(f"train label0 precision", round((train_0precis / iteration()).item(),5), epoch)
    writer.add_scalar(f"train label1 recall", round((train_1recall / iteration()).item(),5), epoch)
    writer.add_scalar(f"train label0 recall", round((train_0recall / iteration()).item(),5), epoch)
    writer.add_scalar(f"train accurancy", round((train_acc / iteration()).item(),5), epoch)
    writer.add_scalar(f"train epoch loss", train_sum_loss, epoch)
    print("  train label1 recall: {:<7}, \ttrain label0 recall: {:<7}, \ttrain loss： {:<7}".format(round((train_1recall / iteration()).item(),5),round((train_0recall/iteration()).item(),5),round(train_sum_loss,5)))
 
    # test the data
    model.eval()
    with torch.no_grad():
        test_sum_loss = 0
        test_1recall  = 0
        test_0recall  = 0
        test_1precis  = 0
        test_0precis  = 0
        test_acc      = 0
        test_recall   = 0

        count = 0
        for i in range(int(y_test.size(0))):
            indices = torch.tensor([i])
            inputs  = torch.index_select(inputs_test, 0, indices).to(device)
            y       = torch.index_select(y_test, 0, indices).to(device)
            outputs = model(inputs).to(device)
            test_loss = loss_fn(outputs, y)
            test_sum_loss += torch.sum(test_loss).item()
            test_TP = torch.where(torch.gt(outputs,threshold),torch.where(torch.gt(outputs,threshold)==y,1,0),0).sum()
            test_FP = torch.where(torch.gt(outputs,threshold),torch.where(torch.gt(outputs,threshold)!=y,1,0),0).sum()
            test_TN = torch.where(torch.le(outputs,threshold),torch.where(torch.gt(outputs,threshold)==y,1,0),0).sum()
            test_FN = torch.where(torch.le(outputs,threshold),torch.where(torch.gt(outputs,threshold)!=y,1,0),0).sum()
            test_1recall += (test_TP+1e-5)/(test_TP+test_FN+1e-5)
            test_0recall += (test_TN+1e-5)/(test_TN+test_FP+1e-5)
            test_1precis += (test_TP+1e-5)/(test_TP+test_FP+1e-5)
            test_0precis += (test_TN+1e-5)/(test_TN+test_FN+1e-5)
            test_acc     += (test_TP+test_TN+1e-5)/(test_TP+test_FP+test_TN+test_FN+1e-5)

    scheduler.step()
    writer.add_scalar(f"test label1 precision", round((test_1precis / y_test.size(0)).item(),5), epoch)
    writer.add_scalar(f"test label0 precision", round((test_0precis / y_test.size(0)).item(),5), epoch)
    writer.add_scalar(f"test label1 recall", round((test_1recall / y_test.size(0)).item(),5), epoch)
    writer.add_scalar(f"test label0 recall", round((test_0recall / y_test.size(0)).item(),5), epoch)
    writer.add_scalar(f"test accurancy", round((test_acc / y_test.size(0)).item(),5), epoch)
    writer.add_scalar(f"test epoch loss", test_sum_loss, epoch)
    print("  test  label1 precis: {:<7}, \ttest  label0 precis: {:<7}, \ttest  loss： {:<7}".format(round((test_1precis / y_test.size(0)).item(),5),round((test_0precis / y_test.size(0)).item(),5),round(test_sum_loss,5)))
    print("  test  label1 recall: {:<7}, \ttest  label0 recall: {:<7}".format(round((test_1recall / y_test.size(0)).item(),5),round((test_0recall/y_test.size(0)).item(),5),round(test_sum_loss,5)))
    print("  train accurancy:     {:<7}, \ttest  accurancy:     {:<7}".format(round((train_acc / iteration()).item(),5), round((test_acc / y_test.size(0)).item(),5)))
        
    model_save_path = save_file+"/model_GeomP_"+str(epoch)+".pt"
    torch.save(model.state_dict(), model_save_path) 