from skimage import io
import numpy as np
import os
import json

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms



class Data(Dataset):
    def __init__(self,data_dir,imgs):
        self.images = imgs
        self.dir = data_dir
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        image = self.images[index]
        image_name = image
        image = plt.imread(os.path.join(self.dir,'{}.jpg'.format(image)))
        return image_name,image
    
def main(final_dir,processed_dir,json_file):
    f = open(json_file)
    labels = json.load(f)
    images_all = list(labels.keys())
    processed_images = os.listdir(processed_dir)
    data = Data(r'D:\RRC.CVC\ArT\train_images',images_all)
    loader = torch.utils.data.DataLoader(data,batch_size = 1,shuffle=1,num_workers=4)
    
    for i,(image_name,img) in enumerate(loader):
        print(image_name[0])
        temp = [s for s in processed_images if ('{}_'.format(image_name[0])) in s]
        #print(temp)
        img = img[0]
        final_image = np.zeros_like(img)
        final_image = np.squeeze(final_image)
        try:
            final_image = np.squeeze(final_image[:,:,0])
        except:
            pass
        for image in temp:
            sub_image = plt.imread(os.path.join(processed_dir,image))
            try:
                sub_image = np.squeeze(sub_image[:,:,0])
            except:
                pass
            x_indices,y_indices = np.where(sub_image!=0)
            if x_indices.size==0 or y_indices.size==0:
                continue
            val = sub_image[x_indices[0],y_indices[0]]
            final_image[x_indices,y_indices] = val
        x,y = np.where(final_image == 0)
        final_image[x,y] = 255
        plt.imsave(os.path.join(final_dir,image_name[0]),final_image,cmap='gray')
        
            
        
if __name__ == '__main__':
    final_image_dir = r'D:\RRC.CVC\ArT\final_images'
    processed_images = r'D:\RRC.CVC\ArT\processed_images'
    json_file = r'D:\RRC.CVC\ArT\train_labels.json'
    if not os.path.exists(final_image_dir):
        os.mkdir(final_image_dir)
        
    main(final_image_dir,processed_images,json_file)
        