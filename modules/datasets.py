import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
from torchvision.transforms.transforms import RandomHorizontalFlip
import argparse
import time
from PIL import Image
from torchvision.utils import save_image

#folders = {'L':"../left", 'R':"../right", 'L_SEG':"../left_seg", 'R_SEG':"../right_seg"}
infos = 'full_5fold_test_v25.csv'
LABELS = ['WHO-CVD_norm']

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

WIDTH = [350,30,22,8,30,1,1,1]
IMAGE_FOLDER = '/home/zlin/wp'
class UkbDataset(Dataset):
    "Dataset include dataset info loading and tokenizing"
    def __init__(self, args, mode, split):
        df = pd.read_csv(infos)
        # 0.5 normal
        # inception 400, 299, 
        # vit rs, 512, 384
        self.transform_aug = transforms.Compose([       
                                transforms.Resize(512),
                                transforms.CenterCrop(512),
                                transforms.ToTensor(),
                                transforms.RandomResizedCrop(384, scale=(0.5, 1.)),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                #transforms.RandomHorizontalFlip(0.5),
                                #transforms.RandomVerticalFlip(0.3),
                                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
                                ])
        self.transform_val = transforms.Compose([          
                                transforms.Resize(384),
                                transforms.CenterCrop(384),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
                                ])
        
        self.transform_Hflip = transforms.Compose([transforms.RandomHorizontalFlip(1)])
        #label = 'WHO-CVD'
        self.mode = mode
    
        if mode == 'train':
            df = df.loc[(df['fold']!=split) & (df['fold']!=5)].copy()
        elif mode == 'val':
            df = df.loc[df['fold']==split].copy()
        elif mode == 'test':
            df = df.loc[df['fold']==5].copy()
        print("LOADING DATA SHAPE:",df.shape)
        
        self.items = []
        for idx, row in df.iterrows():
            left = os.path.join(IMAGE_FOLDER,'left',row['left'])
            right = os.path.join(IMAGE_FOLDER, 'right',row['right'])
            #if os.path.exists(img) and os.path.exists(seg): already checked in data_pre
            
            levels = []
            for id_tar, tar in enumerate(LABELS):
                num_classes = WIDTH[id_tar] 
                label = int(row[tar])
                levels = levels + [1]*label + [0]*(num_classes - label)        
            self.items.append({'idx':idx,
                            'side':row['filecheck'],
                            'left':left,
                            'right':right,
                            'target':np.array([row[x] for x in LABELS],dtype=np.float32),
                            'ordi_target':np.array(levels,dtype=np.float32),
                            })

        print("{} split length:{}".format(mode,len(self.items)))  
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):       
        item = self.items[idx]
        side = item['side']
        imgs = []
        
        if self.mode == 'train':
            trans = self.transform_aug
        else:
            trans = self.transform_val
            
        if side == 1:
            # left_only, use another agumentation as right
            image = Image.open(item['left']).convert('RGB')
            img = trans(image)
            imgs.append(img)
            imgs.append(self.transform_Hflip(img))
            # the mode is only for validate
            amount = 1 
            
        elif side == 2:
            # right_only
            image = Image.open(item['right']).convert('RGB')
            img = trans(image)
            imgs.append(img)
            imgs.append(self.transform_Hflip(img))
            amount = 1
        else:
            # both eye
            image = Image.open(item['left']).convert('RGB')
            img = trans(image)
            imgs.append(img)
            
            image = Image.open(item['right']).convert('RGB')
            img = trans(image)
            imgs.append(img)
            amount = 2       

        return item['idx'], torch.stack(imgs), amount, item['target'], item['ordi_target']
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retina Health Score Training')
    args = parser.parse_args()

    #train = db_UkbDataset(args,'train')
    val = UkbDataset(args,'val',4)
    
    _,img,mask,target,ordi_target = val.__getitem__(0)
    print(mask)
    print(target)
    
    print(ordi_target.shape)
