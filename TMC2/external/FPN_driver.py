#!/usr/bin/env python
# coding: utf-8
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

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

# def model_init(path_GeomI, path_GeomP, path_AttrI, path_AttrP): 
def model_init(path_GeomP, path_AttrI): 
    # global model_GeomI
    global model_GeomP  # FPN_GeomP
    global model_AttrI  # FPN_AttrI
    # global model_AttrP

    # model_GeomI = CFPN().cuda()
    model_GeomP = CFPN().cuda()
    model_AttrI = CFPN().cuda()
    # model_AttrP = CFPN().cuda()

#     try:
#         model_GeomI.load_state_dict(torch.load(path_GeomI, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
# #        model_GeomI.load_state_dict(torch.load(path_GeomI, map_location=torch.device("cpu")))
#     except Exception as result:
#         print("MesksMethodError::",result)

    try:
        model_GeomP.load_state_dict(torch.load(path_GeomP, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
#        model_GeomP.load_state_dict(torch.load(path_GeomP, map_location=torch.device("cpu")))
    except Exception as result:
        print("MesksMethodError::",result)

    try:
        model_AttrI.load_state_dict(torch.load(path_AttrI, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
#        model_AttrI.load_state_dict(torch.load(path_AttrI, map_location=torch.device("cpu")))
    except Exception as result:
        print("MesksMethodError::",result)

#     try:
#         model_AttrP.load_state_dict(torch.load(path_AttrP, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
# #        model_AttrP.load_state_dict(torch.load(path_AttrP, map_location=torch.device("cpu")))
#     except Exception as result:
#         print("MesksMethodError::",result)

    # model_GeomI.eval()
    model_GeomP.eval()
    model_AttrI.eval()
    # model_AttrP.eval()

def nnDriver(BaseFeatures, OccuFeatures, QP, model_type):
    try:
        BaseFeatures = torch.tensor(BaseFeatures,dtype=torch.float32).reshape(1,64,64).cuda()
        OccuFeatures = torch.tensor(OccuFeatures,dtype=torch.float32).reshape(1,64,64).cuda()
        qp           = (torch.ones((64,64))*QP).reshape(1,64,64).cuda()
        input_sample = torch.stack([BaseFeatures,OccuFeatures,qp],axis=3).permute(0,3,2,1).cuda()

        output=[]
        if model_type==0:
        #   output = model_GeomI(input_sample).tolist()
            output = model_GeomI(input_sample).tolist()
        elif model_type==1:
        #   output = model_GeomP(input_sample).tolist()
            output = model_GeomP(input_sample).tolist()
        elif model_type==2:
        #   output = model_AttrI(input_sample,QP).tolist()
          output = model_AttrI(input_sample).tolist()
        elif model_type==3:
        #   output = model_AttrP(input_sample).tolist()
            output = model_AttrP(input_sample).tolist()

        # print(output)
    
    except Exception as result:
        print("MesksMethodError::",result)

    return output


if __name__ == '__main__':
    main()