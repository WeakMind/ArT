# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:55:20 2019

@author: h.oberoi
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import json

data = {}

data_dir = r'D:\RRC.CVC\ArT\processed_images_2'
l = os.listdir(data_dir)
for i,img_name in enumerate(l):
    #print(img_name)
    img = plt.imread(os.path.join(data_dir,img_name))
    data[img_name] = []
    img = np.array(img*255).astype(int)
    img[np.where(img==255)] = 0
    img = img.flatten()
    img_unique = np.unique(img)
    data[img_name].append({})
    for val in img_unique:
        count = np.where(img==val)[0].size
        data[img_name][0][str(val)] = count
        
with open('data.json','w') as outputfile:
    json.dump(data,outputfile)