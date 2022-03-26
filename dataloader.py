from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io as scio
import torch

class Mydataset(Dataset):
 #传递数据路径，csv路径 ，数据增强方法
    def __init__(self, dataPath, istest=False, transform=None, target_transform=None):
        super(Mydataset, self).__init__()
        #一个个往列表里面加绝对路径
        self.path = []
        #读取csv
    
        self.data = np.array(scio.loadmat(dataPath)['data'])
        self.x = self.data[:,0:71]
        self.y = self.data[:,71]
        
        
     #  最关键的部分，在这里使用前面的方法
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        y = np.array([y], dtype=np.double)
        
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        
        return x, y
    def __len__(self):
        return len(self.data)


