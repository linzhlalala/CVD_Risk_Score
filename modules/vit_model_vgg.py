from __future__ import print_function, division, absolute_import
import torch
from torch.functional import Tensor
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import timm 

class vit_mt(nn.Module):
    def __init__(self, opt=None):
        super(vit_mt, self).__init__()

        self.img_model = timm.create_model('vgg16', num_classes=0,pretrained=True)
        feature_dim = 384
        
        self.fc = nn.Sequential(nn.Linear(feature_dim, 443))

    def forward(self, img):
        # batch
        #B = img.shape[0]
        # reshape
        x = torch.flatten(img, start_dim=0, end_dim=1)
        # features
        z = self.img_model(x)
        # predict
        preds = self.fc(z)
        #print(z)
        return z, preds
    

if __name__ == '__main__':
    model =  vit_mt()
    print('success')

    img = torch.rand(8,4,3,384,384)
    x, res = model(img)
    print(x.shape, res.shape)
