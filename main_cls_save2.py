#this edition is for single target regression
import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from modules.datasets import UkbDataset
from modules.loss_function import Mix_Loss
from modules.vit_model import vit_mt
#from modules.loss_function import weighted_loss
import warnings
warnings.filterwarnings(action='once')

parser = argparse.ArgumentParser(description='Retina Health Score Training')
parser.add_argument('--start-epoch', default=0, type=int, 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, 
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--print-freq', default=50, type=int,
                    help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=1024, type=int,
                    help='seed for initializing training. 1024 for reproductivity')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')  
#debugging args, change it on other device
parser.add_argument('--workers', default=4, type=int, 
                    help='number of data loading workers (default: 0)')
parser.add_argument('--batch-size', default=32 , type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', default=100, type=int, 
                    help='number of total epochs to run')
parser.add_argument('--id', default='rgs', type=str, 
                    help='job id to save logs')

checkpoint_path = ''


LABELS = ['WHO-CVD'] # BCE
    
WIDTH = [350,30,22,8,30,1,1,1]
label_pos = [0,350,380,402,410,440,441,442,443]
rgs_divider = 5
    
def main():
    global checkpoint_path
    args = parser.parse_args()
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)       
        np.random.seed(args.seed)

        cudnn.deterministic = True
        print("Using seed : {args.seed}".format)

    checkpoint_path = 'logs/'+args.id

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    loss = Mix_Loss().cuda(args.gpu)

    best_acc_10 = -np.inf
    best_acc_20 = -np.inf

    log_file = os.path.join(checkpoint_path,"logs.csv")
    if os.path.exists(log_file):
        os.rename(log_file,log_file.replace(".csv","_{}.csv".format(time.time())))

    model = vit_mt(args).cuda(args.gpu)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=False)
    cudnn.benchmark = True
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch'] + 1
            best_acc_10 = checkpoint['best_acc_10']
            best_acc_20 = checkpoint['best_acc_20']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # Data loading and split it into train, val, test

    train_dataset = UkbDataset(args,'train',4)
    val_dataset = UkbDataset(args,'val',4)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,num_workers=args.workers, pin_memory=True, shuffle = True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,num_workers=args.workers, pin_memory=True, shuffle = False)
    # train val and test
    # acc1 actually is mse(l2) at here
    for epoch in range(args.start_epoch, args.epochs):
        # no adjust as there is plateau scheduler instead
        # adjust_learning_rate(optimizer, epoch, args)

        end = time.time()
        # train for one epoch
        train_loss = train(train_loader, model, loss, optimizer, epoch, args)

        val_loss, val_score_dict  = validate(val_loader, model, loss, epoch, args)
        
        val_score = val_score_dict['WHO-CVD_100_AUC']
        is_best_10 = val_score > best_acc_10
        best_acc_10 = max(val_score, best_acc_10)

        val_score = val_score_dict['WHO-CVD_200_AUC']
        is_best_20 = val_score > best_acc_20
        best_acc_20 = max(val_score, best_acc_20)

        # Update info to a logging csv
        time_cost = time.time() - end

        epoch_dict = {"epoch":epoch, "train_loss":train_loss, "val_loss":val_loss,
                    "epoch_time":time_cost, "lr": optimizer.param_groups[0]['lr']}
        epoch_dict.update(val_score_dict)

        log_to_file(log_file,epoch_dict)
        # checkpoint
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc_10': best_acc_10,
                'best_acc_20': best_acc_20,
                #'optimizer' : optimizer.state_dict(),
            }
        torch.save(state, os.path.join(checkpoint_path,'checkpoint.pth.tar'))
        #best
        if is_best_10:
            torch.save(state, os.path.join(checkpoint_path,'model_best_10.pth.tar'))
        if is_best_20:
            torch.save(state, os.path.join(checkpoint_path,'model_best_20.pth.tar'))
        # renew lr
        scheduler.step(val_loss)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    
    end = time.time()
    for i, (_, img, _, _, ordi_target) in enumerate(train_loader):
        # measure data loading time
        if args.gpu is not None:
            img = img.cuda(args.gpu, non_blocking=True)
            #mask = mask.cuda(args.gpu, non_blocking=True)
            ordi_target = ordi_target.cuda(args.gpu, non_blocking=True).to(torch.float32)
        # compute output
        _, pred = model(img)
        #print(output.shape, target.shape)
        loss = criterion(pred, ordi_target)
        # measure accuracy and record loss
        #acc1 = accuracy(output, target)
        losses.update(loss.item(), img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1) % args.print_freq == 0:
            progress.display(i)

    return losses.avg

def validate(val_loader, model, criterion,  epoch, args):
    global checkpoint_path

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time,losses],
        prefix='Val/Test: ')

    # switch to evaluate mode
    model.eval()

    idxs = []
    gts_od = []
    gts = []
    preds = []
    with torch.no_grad():
        end = time.time()
        for i, (idx, img, modes, target, ordi_target) in enumerate(val_loader):
            if args.gpu is not None:
                img = img.cuda(args.gpu, non_blocking=True)
                #mask = mask.cuda(args.gpu, non_blocking=True)
                ordi_target = ordi_target.cuda(args.gpu, non_blocking=True).to(torch.float32)

            _, pred = model(img)
            loss = criterion(pred, ordi_target)
            losses.update(loss.item(), img.size(0))

            # reg-like targets
            pred = torch.sigmoid(pred).detach().cpu().numpy()
            target = target.cpu().numpy()
            ordi_target = ordi_target.cpu().numpy()
            idx = idx.cpu().numpy()
            for i_mode, mode in enumerate(modes):
                preds.append(pred[2*i_mode:2*i_mode+mode,])                  
                gts.append(np.expand_dims(target[i_mode,],axis = 0).repeat(mode,axis=0))   
                gts_od.append(np.expand_dims(ordi_target[i_mode,],axis = 0).repeat(mode,axis=0))   
                idxs.append(idx[i_mode].repeat(mode,axis=0))
             
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if (i+1) % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        #print(' *VAL R2 {losses.avg:.3f}'.format(losses=losses))
    index_arr = np.concatenate(idxs,axis=0).reshape(-1,1)
    gts_arr = np.concatenate(gts,axis=0)
    gts_od_arr= np.concatenate(gts_od, axis=0)
    preds_arr= np.concatenate(preds, axis=0)
    
    columns = []
    for i,label in enumerate(LABELS):
        columns += [f"{label}_{x+1}" for x in range(WIDTH[i])]
    
    cols = ['idx'] + columns + [x+'_p' for x in columns] + LABELS
    # print(len(cols))
    # print(np.concatenate((index_arr,gts_od_arr,preds_arr,gts_arr),axis=1).shape)
    df = pd.DataFrame(data = np.concatenate((index_arr,gts_od_arr,preds_arr,gts_arr),axis=1),columns = cols)
    df.to_csv(os.path.join(checkpoint_path,"res_{}.csv".format(epoch)), index = False)

    #r2 = {col+'_r2':r2_score(df[col], df[col+'_p']) for col in ['WHO-CVD','Age']}
    #mae = {col+'_mae':mean_absolute_error(df[col], df[col+'_p']) for col in ['WHO-CVD','Age']}
    metris = {}
    for x in columns:
        try: 
            metris[f"{x}_AUC"] = roc_auc_score(df[x], df[x+'_p'])
        except:
            pass
    return losses.avg, metris


def log_to_file(log_file,epoch_dict):
    """log training infomation to csv for better display"""
    #global checkpoint_path
    #log_file = os.path.join(checkpoint_path,"logs.csv")

    crt_time = time.asctime(time.localtime(time.time()))
    epoch_dict['time'] = crt_time

    if not os.path.exists(log_file):
        record_table = pd.DataFrame()
    else:
        record_table = pd.read_csv(log_file)
    record_table = record_table.append(epoch_dict, ignore_index=True)
    record_table.to_csv(log_file, index=False)

def save_checkpoint(state, is_best, filename):
    #current
    global checkpoint_path
    torch.save(state, os.path.join(checkpoint_path,filename))
    #best
    if is_best:
        shutil.copyfile(os.path.join(checkpoint_path,filename),os.path.join(checkpoint_path,filename.replace('checkpoint','model_best')))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()