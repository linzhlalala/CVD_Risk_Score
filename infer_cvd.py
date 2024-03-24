#based on backbone model_484_fold4(UKB-WHO), DA_v0.2(GZ+IN+MEL)
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from model.vit_model import vit_mt
from ranker import get_percentile

device = torch.device("cpu")
mpath = './model/cvd_254_rss20.pth.tar'
state_dict = torch.load(mpath, map_location='cpu') 

model = vit_mt()
model.load_state_dict(state_dict['state_dict'], strict=1)
model.eval()
model.to(device)

trans = transforms.Compose([      
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),   
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

trans_hflip = transforms.Compose([            
    transforms.RandomHorizontalFlip(1)
    ])

ukb_thresholds = np.load("model/254_rss20_thh.npy")

def argwhere(arr):
    a = np.where(arr == 0)[0]
    if  len(a) != 0:
        return a[0]
    else:
        return 350
## Basic, standard thresholds
def cvd_v1(org_img_path, camera = None, age = 50, gender = 1):
    org = Image.open(org_img_path)
    
    img = trans(org).unsqueeze(0)
    thh = ukb_thresholds[9]
    with torch.no_grad():
        _, pred = model(img)
        p1 = torch.sigmoid(pred).detach().cpu().numpy()
        p1 = p1 > thh
    cvd_score = np.apply_along_axis(argwhere, 1, p1)
    cvd_score = cvd_score.mean()/10.0
    percentile = get_percentile(cvd_score, age = age, gender = gender)
    return cvd_score, percentile

## With a optimal thresholds
def cvd_v1_opt(org_img_path, camera = None, age = 50, gender = 1):
    org = Image.open(org_img_path)
    
    img = trans(org).unsqueeze(0)
    with torch.no_grad():
        _, pred = model(img)
        if camera == 'mw':
            thh = ukb_thresholds[7]
            p1 = torch.sigmoid(pred).detach().cpu().numpy()
            p1 = p1 > thh
        else:
            thh = ukb_thresholds[9]
            p1 = torch.sigmoid(pred).detach().cpu().numpy()
            p1 = p1 > thh
    cvd_score = np.apply_along_axis(argwhere, 1, p1)
    cvd_score = cvd_score.mean()/10.0
    percentile = get_percentile(cvd_score, age = age, gender = gender)
    return cvd_score, percentile

if __name__ == '__main__':
    import time
    t0 = time.time()
    print(cvd_v1('jason-test/1_od.jpg','mw', age = 45, gender = 1))    
    print(cvd_v1('jason-test/1_os.jpg','mw', age = 45, gender = 0))
    print(cvd_v1('jason-test/1_od.jpg','tc', age = 45, gender = 1))    
    print(cvd_v1('jason-test/1_os.jpg','tc', age = 45, gender = 0))

    print(cvd_v1_opt('jason-test/1_od.jpg','mw', age = 45, gender = 1))    
    print(cvd_v1_opt('jason-test/1_os.jpg','mw', age = 45, gender = 0))
    print(cvd_v1_opt('jason-test/1_od.jpg','tc', age = 45, gender = 1))    
    print(cvd_v1_opt('jason-test/1_os.jpg','tc', age = 45, gender = 0))
    print(f"spend: {time.time() -  t0} s")

# (2.4, 0.9455442373868148)
# (2.4, 0.9078027172391672)
# (2.4, 0.9455442373868148)
# (2.4, 0.9078027172391672)
# (2.0, 0.9473283788710045)
# (1.9, 0.9144136953865473)
# (2.4, 0.9455442373868148)
# (2.4, 0.9078027172391672)
# spend: 8.287595748901367 s