
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class conv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding):
        super(conv2d,self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = kernel_size,padding = padding)
        self.batch = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self,X):
        X = self.conv(X)
        X = self.batch(X)
        X = self.relu(X)
        return X

class conv_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,codeword):
        super(conv_block,self).__init__()
        self.conv1 = conv2d(in_channels,out_channels,kernel_size,padding)
        if codeword == 'C':
            self.conv2 = conv2d(out_channels,out_channels,kernel_size,padding)
        else:
            self.conv2 = conv2d(out_channels,out_channels//2,kernel_size,padding)
        
    def forward(self,X):
        X = self.conv1(X)
        X = self.conv2(X)
        return X
    
class up_sample(nn.Module):
    def __init__(self,in_,out_):
        super(up_sample,self).__init__()
        self.convT = nn.ConvTranspose2d(in_channels= in_,out_channels = out_ ,padding = (1,1),kernel_size= (4,4),stride = (2,2))
        self.batch_norm = nn.BatchNorm2d(in_)
        self.relu = nn.ReLU()
        
    def forward(self,X):
        X = self.convT(X)
        X = self.batch_norm(X)
        X = self.relu(X)
        return X

class down_sample(nn.Module):
    def __init__(self):
        super(down_sample,self).__init__()
        
    def forward(self,X):
        return F.max_pool2d(X,kernel_size = (2,2))
        
class Model(nn.Module):
    def __init__(self,boolean):
        super(Model,self).__init__()
        self.model = models.vgg16_bn(pretrained=boolean)
        self.model.load_state_dict(torch.load('D:/vgg16_bn-6c64b313.pth'))
        self.model = self.model.features;
        
        
        self.conv_block1 = self.model[:6]
        self.conv_block2 = self.model[7:13]
        self.conv_block3 = self.model[14:23]
        self.conv_block4 = self.model[24:33]
        self.conv_block5 = self.model[34:43]
        self.up_512_1 = up_sample(512,512)
        self.conv_block6 = conv_block(1024,512,(3,3),(1,1),'D')
        self.up_512_2 = up_sample(256,256)
        self.conv_block7 = conv_block(512,256,(3,3),(1,1),'D')
        self.up_256 = up_sample(128,128)
        self.conv_block8 = conv_block(256,128,(3,3),(1,1),'D')
        self.up_128 = up_sample(64,64)
        self.conv_block9 = conv_block(128,64,(3,3),(1,1),'C')
        
        self.down_sample = down_sample()
        self.upsample = nn.Upsample(scale_factor=2)
        
        
        self.text = nn.Conv2d(in_channels = 64,out_channels = 2,kernel_size = (1,1))
        
        self.left_top = nn.Conv2d(in_channels = 64,out_channels = 2,kernel_size = (1,1))
        
        self.top = nn.Conv2d(in_channels = 64,out_channels = 2,kernel_size = (1,1))
        
        self.right_top = nn.Conv2d(in_channels = 64,out_channels = 2,kernel_size = (1,1))
        
        self.right = nn.Conv2d(in_channels = 64,out_channels = 2,kernel_size = (1,1))
        
        self.right_down = nn.Conv2d(in_channels = 64,out_channels = 2,kernel_size = (1,1))
        
        self.down = nn.Conv2d(in_channels = 64,out_channels = 2,kernel_size = (1,1))
        
        self.left_down = nn.Conv2d(in_channels = 64,out_channels = 2,kernel_size = (1,1))
        
        self.left = nn.Conv2d(in_channels = 64,out_channels = 2,kernel_size = (1,1))
        
        
        
        
          
    def forward(self,X):
        
        #X = self.model(X)
        
        first = self.conv_block1(X)
        #import pdb;pdb.set_trace()
        X = self.down_sample(first)
        #import pdb;pdb.set_trace()
        second = self.conv_block2(X)
        
        X = self.down_sample(second)
        
        third = self.conv_block3(X)
        
        X = self.down_sample(third)
        
        fourth = self.conv_block4(X)
        X = self.down_sample(fourth)
        fifth = self.conv_block5(X)
        
        X = self.up_512_1(fifth)
        
        sixth = torch.cat((X,fourth),dim = 1)
        sixth = self.conv_block6(sixth)
        X = self.upsample(sixth)
        seventh = torch.cat((X,third),dim = 1)
        seventh = self.conv_block7(seventh)
        X = self.upsample(seventh)
        eigth = torch.cat((X,second),dim = 1)
        eigth = self.conv_block8(eigth)
        X = self.up_128(eigth)
        ninth = torch.cat((X,first),dim= 1)
        ninth = self.conv_block9(ninth)
        
        
        text = self.text(ninth)
        left_top = self.left_top(ninth)
        top = self.top(ninth)
        right_top = self.right_top(ninth)
        right = self.right(ninth)
        right_down = self.right_down(ninth)
        down = self.down(ninth)
        left_down = self.left_down(ninth)
        left = self.left(ninth)
        
        return {'out' : (text,left_top,top,right_top,right,right_down,down,left_down,left)}