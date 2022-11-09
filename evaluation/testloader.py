# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 21:40:11 2022

@author: 80594
"""
import os
import pandas as pd
import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms
import kornia as K
Transform112 = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor(),
    ])
base_transforms = transforms.Compose([
    transforms.Resize([112,112]),
    transforms.ToTensor(),
    transforms.Lambda(lambda img:K.color.rgb_to_luv(img))
    ])
Transform227 = transforms.Compose([
    transforms.Resize([227, 227]),
    transforms.ToTensor(),
    ])
class GTSRB_Test_Loader(Dataset):
    '''
    TEST_PATH:
        should be the path you reserve your test image. For example 'GTSRB/Final_Test/Images/'
    TEST_GT_PATH
        
    '''
    def __init__(self, TEST_PATH = None, TEST_GT_PATH = 'evaluation/GTSRB_Test_GT.csv', MODEL=112):
        self.df = pd.read_csv(TEST_GT_PATH,sep=';')
        self.TEST_PATH = TEST_PATH
        if (MODEL == 112):
            self.Transform = base_transforms
        else:
            self.Transform = Transform227
        self.MODEL = MODEL

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filename = os.path.join(self.TEST_PATH, row['Filename'])
        img = Image.open(filename)
        img = self.Transform(img)
        
        label = int(row['ClassId'])
        return img, label, filename


