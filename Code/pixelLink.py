# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:47:09 2019

@author: h.oberoi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
import matplotlib.pyplot as plt
import json
from pprint import pprint

import torchvision
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset
import random
import numpy as np

from Model import Model
from skimage import util

class Data(Dataset):
    def __init__(self,train_dir,groundtruth):
        self.train_dir = train_dir
        self.groundtruth = groundtruth
        
        self.images = os.listdir(self.train_dir)
        
    def __len__(self):
        return len(self.images)
    
    def transform(self,x,y,image_name):
        
        y = np.expand_dims(y,axis=0)
        y = np.transpose(np.vstack((y,y,y)),(1,2,0))
        y = np.dot(y,255).astype('int32')
        y = y.astype('uint8')
        
        x = transforms.ToPILImage()(x)
        y = transforms.ToPILImage()(y)
        
        
        try:
            i,j,k,w = transforms.RandomCrop.get_params(x,[224,224])
        except:
            i,j,k,w = 0,0,x.shape[0],x.shape[1]
        
        x = TF.crop(x,i,j,k,w)
        y = TF.crop(y,i,j,k,w)
        
        #y = transforms.Resize((400,400))(y)
        
        print(np.unique(y))
        
        if random.random() > 0.5:
            x = transforms.RandomHorizontalFlip(p=1)(x)
            y = transforms.RandomHorizontalFlip(p=1)(y)
        
        angle = random.uniform(-20,20)
        
        
        if random.random() <= 0.2:
            x = transforms.RandomRotation((angle,angle))(x)
            y = transforms.RandomRotation((angle,angle))(y)
        
        x = transforms.ToTensor()(x)
        y = np.array(y)
        
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])(x)
        y = np.squeeze(y[:,:,0])
        y[np.where(y==255)] = 0
        
        text = np.zeros_like(y)
        text[np.where(y!=0)] = 1
        
        left_top = np.zeros_like(y)
        top = np.zeros_like(y)
        right_top = np.zeros_like(y)
        right = np.zeros_like(y)
        right_down = np.zeros_like(y)
        down = np.zeros_like(y)
        left_down = np.zeros_like(y)
        left = np.zeros_like(y)
        
        file = open('data.json','r')
        data = json.load(file)
        #import pdb;pdb.set_trace()
        S= sum(data['{}.png'.format(image_name)][0].values())
        N = len(data['{}.png'.format(image_name)][0].keys())
        
        B = S/N
        
        text_coeff = np.zeros_like(text)
        
        
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                
                text_coeff[i,j] = (B/data['{}.png'.format(image_name)][0][str(y[i,j])])
                text_coeff[np.where(text_coeff==0)] = 1
                
                if i-1 >=0 and j-1 >=0 and y[i-1,j-1]==y[i,j]:
                    left_top[i,j] = 1
                else:
                    left_top[i,j] = 0
                
                if i-1 >=0  and y[i-1,j]==y[i,j]:
                    top[i,j] = 1
                else:
                    top[i,j] = 0
                    
                if i-1 >=0 and j+1 < y.shape[1] and y[i-1,j+1]==y[i,j]:
                    right_top[i,j] = 1
                else:
                    right_top[i,j] = 0
                    
                if j+1 < y.shape[1] and y[i,j+1]==y[i,j]:
                    right[i,j] = 1
                else:
                    right[i,j] = 0
                    
                if i+1 < y.shape[0] and j+1 < y.shape[1] and y[i+1,j+1]==y[i,j]:
                    right_down[i,j] = 1
                else:
                    right_down[i,j] = 0
                    
                if i+1 < y.shape[0] and y[i+1,j]==y[i,j]:
                    down[i,j] = 1
                else:
                    down[i,j] = 0
                    
                if i+1 < y.shape[0] and j-1 >=0 and y[i+1,j-1]==y[i,j]:
                    left_down[i,j] = 1
                else:
                    left_down[i,j] = 0
                    
                if  j-1 >=0 and y[i,j-1]==y[i,j]:
                    left[i,j] = 1
                else:
                    left[i,j] = 0
                    
        import pdb;pdb.set_trace()            
        return x,{'out':(text,left_top,top,right_top,right,right_down,down,left_down,left)}
    
    def __getitem__(self,index):
        img = self.images[index]
        image_name = img.split('.')[0]
        x = plt.imread(os.path.join(self.train_dir,img))
        y = plt.imread(os.path.join(self.groundtruth,'{}.png'.format(image_name)))
        
        x,y = self.transform(x,y,image_name)
        
        return img,x,y


def calculate_loss(out,out_y,loss_function):
    out = out['out']
    text_x,left_top_x,top_x,right_top_x,right_x,right_down_x,down_x,left_down_x,left_x = out
    out_y = out_y['out']
    text,left_top,top,right_top,right,right_down,down,left_down,left = out_y
    
    b,c,l,w = text_x.shape
    text_x = text_x.view(b,c,-1).permute(0,2,1).view(-1,c)
    
    b,c,l,w = left_top_x.shape
    left_top_x = left_top_x.view(b,c,-1).permute(0,2,1).view(-1,c)
    
    b,c,l,w = top_x.shape
    top_x = top_x.view(b,c,-1).permute(0,2,1).view(-1,c)
    
    b,c,l,w = right_top_x.shape
    right_top_x = right_top_x.view(b,c,-1).permute(0,2,1).view(-1,c)
    
    b,c,l,w = right_x.shape
    right_x = right_x.view(b,c,-1).permute(0,2,1).view(-1,c)
    
    b,c,l,w = right_down_x.shape
    right_down_x = right_down_x.view(b,c,-1).permute(0,2,1).view(-1,c)
    
    b,c,l,w = down_x.shape
    down_x = down_x.view(b,c,-1).permute(0,2,1).view(-1,c)
    
    b,c,l,w = left_down_x.shape
    left_down_x = left_down_x.view(b,c,-1).permute(0,2,1).view(-1,c)
    
    b,c,l,w = left_x.shape
    left_x = left_x.view(b,c,-1).permute(0,2,1).view(-1,c)
    
    X = torch.cat((left_top_x,top_x,right_top_x,right_x,right_down_x,down_x,left_down_x,left_x),0)
    
    b,l,w = text.shape
    text = text.view(b,-1).view(-1)
    
    b,l,w = left_top.shape
    left_top = left_top.view(b,-1).view(-1)
    
    b,l,w = top.shape
    top = top.view(b,-1).view(-1)
    
    b,l,w = right_top.shape
    right_top = right_top.view(b,-1).view(-1)
    
    b,l,w = right.shape
    right = right.view(b,-1).view(-1)
    
    b,l,w = right_down.shape
    right_down = right_down.view(b,-1).view(-1)
    
    b,l,w = down.shape
    down = down.view(b,-1).view(-1)
    
    b,l,w = left_down.shape
    left_down = left_down.view(b,-1).view(-1)
    
    b,l,w = left.shape
    left = left.view(b,-1).view(-1)
    
    loss_text = loss_function(text_x,text.long())
    
    Y = torch.cat((left_top,top,right_top,right,right_down,down,left_down,left),0).long()
    
    loss_neighbour = loss_function(X,Y)
    
    return torch.mean(2*loss_text) + torch.mean(loss_neighbour)

def main(train_dir,groundtruth):
    
    epoch = 1000
    
    data = Data(train_dir,groundtruth)
    loader = torch.utils.data.DataLoader(data,batch_size = 1,num_workers=0,shuffle=False)
    
    model = Model(False)
    
    loss_function = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001,weight_decay = 0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,gamma=0.9,step_size=10)
    
    
    for j in range(epoch):
        
        for i,(image_name,x,out_y) in enumerate(loader):
            
            out = model(x)
            loss = calculate_loss(out,out_y,loss_function)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(loss.item())
            
        
        lr_scheduler.step()
        torch.save(model.state_dict(),'./../weights/model_{}.ckpt'.format(epoch))
if __name__ == '__main__':
    train_dir = r'D:\RRC.CVC\ArT\train_images'
    groundtruth = r'D:\RRC.CVC\ArT\processed_images_2'
    main(train_dir,groundtruth)
    



