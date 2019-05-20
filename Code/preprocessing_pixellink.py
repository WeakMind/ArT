# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:02:00 2019

@author: h.oberoi
"""

from skimage import io
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset
import json
from torchvision import transforms

import os
import queue
import numpy as np

class Data(Dataset):
    def __init__(self,data_dir,imgs):
        self.path = data_dir
        self.images = imgs
        
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        img_name = self.images[index]
        name = img_name
        img = plt.imread(os.path.join(self.path,'{}.jpg'.format(img_name)))
        
        
        return name,img
    
def main(processed_data_dir):
    f = open(r'D:\RRC.CVC\ArT\train_labels.json')
    labels = json.load(f)
    l = list()
    print(len(list(labels.keys())))
    for name in list(labels.keys()):
        if name == '':
            continue
        l.append(name)
    
    print(len(l))
    #import pdb;pdb.set_trace()
    
    #io.imshow(r'D:\RRC.CVC\ArT\train_images\gt_3137.jpg')
    #import pdb;pdb.set_trace()
    #t = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    data = Data(r'D:\RRC.CVC\ArT\train_images',l)
    
    data_loader = torch.utils.data.DataLoader(data,batch_size = 1,shuffle=False,num_workers=4)
    

    for i,(img_name,img) in enumerate(data_loader):
        print(img_name[0])
        img = img[0]
        
        for j,d in enumerate(labels[img_name[0]]):
            print(j,img_name[0])
            
            points  =  np.asarray(d['points'])
            if j ==0:
                img = np.zeros_like(img)
            #import pdb;pdb.set_trace()
            cv2.fillPoly(img, pts =[points], color=(j+1,j+1,j+1))
            #print(np.unique(img))
            #import pdb;pdb.set_trace()
            #cv2.imwrite('{}\{}_{}.jpg'.format(processed_data_dir,img_name[0],i),img)
        try:    
            img = np.squeeze(img[:,:,0])
        except:
            pass
        x,y = np.where(img==0)
        img[x,y] = 255
        cv2.imwrite('{}\{}.png'.format(processed_data_dir,img_name[0]),img)
        #import pdb;pdb.set_trace()
        
        
if __name__ == '__main__':
    processed_data_dir = r'D:\RRC.CVC\ArT\processed_images_2'
    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)
    main(processed_data_dir)