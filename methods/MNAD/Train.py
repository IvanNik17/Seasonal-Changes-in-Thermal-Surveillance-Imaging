import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
# from model.utils import DataLoader
from model.habor_utils import DataLoader
from sklearn.metrics import roc_auc_score
from utils import *
import random
from tqdm.autonotebook import tqdm

import argparse
from torch.utils.tensorboard import SummaryWriter
import json


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--val_epoch', type=int, default=5, help='evaluate the model every %d epoch')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--method', type=str, default='pred', choices=['recon', 'pred'], help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='one_day', choices=['one_day', 'one_week', 'one_month'], help='type of dataset: one_day, one_week, one_month')
# parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='exp/log', help='directory of log')
parser.add_argument('--datasplit_dir', type=str, default=r'/home/jacob/code/Seasonal-Changes-in-Thermal-Surveillance-Imaging/splits', help='directory containing datasplits')

args = parser.parse_args()

log_dir = args.exp_dir
if os.path.exists(log_dir):
    raise FileExistsError(log_dir)
else:
    print(f"log_dir: {log_dir}")
    os.makedirs(log_dir)

orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'),'w')
sys.stdout= f

args_file = os.path.join(log_dir, 'train_args.json')
with open(args_file, 'w') as fp:
    json.dump(args.__dict__, fp, indent=2)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

if args.dataset_type == 'one_day':
    train_folder = os.path.join(args.datasplit_dir, 'feb_day_5000.csv')
elif args.dataset_type == 'one_week': 
    train_folder = os.path.join(args.datasplit_dir, 'feb_week_5000.csv')
elif args.dataset_type == 'one_month': 
    train_folder = os.path.join(args.datasplit_dir, 'feb_month_5000.csv')

# Loading dataset
train_dataset = DataLoader('train',train_folder, transforms.Compose([
             transforms.ToTensor(),          
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)
val_dataset = DataLoader('validation', train_folder, transforms.Compose([
             transforms.ToTensor(),          
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
val_batch = data.DataLoader(val_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)


# Model setting
assert args.method == 'pred' or args.method == 'recon', 'Wrong task name'
if args.method == 'pred':
    from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
    img_slice = (args.t_length - 1) * args.c
else:
    from model.Reconstruction import *
    model = convAE(args.c, memory_size = args.msize, feature_dim = args.fdim, key_dim = args.mdim)
params_encoder =  list(model.encoder.parameters()) 
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr = args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
model.cuda()

# Report the training process
train_writer = SummaryWriter(log_dir=log_dir)
loss_func_mse = nn.MSELoss(reduction='none')
m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items

# Training
time_start = time.time()
for epoch in range(1, args.epochs + 1):
    model.train()
    tr_px_loss, tr_cp_loss, tr_sp_loss, tr_tot_loss = 0.0, 0.0, 0.0, 0.0
    progress_bar = tqdm(train_batch)

    for j,(imgs,_) in enumerate(progress_bar):
        progress_bar.set_description(f"Epoch {epoch:3d}/{args.epochs:3d}")

        imgs = Variable(imgs).cuda()
        
        if args.method == 'pred':
            outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs[:,0:img_slice], m_items, True)
        else:
            outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs, m_items, True)
        
        optimizer.zero_grad()
        if args.method == 'pred':
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:,img_slice:]))
        else:
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs))
            
        loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
        loss.backward(retain_graph=True)
        optimizer.step()

        tr_px_loss += loss_pixel.data.item()
        tr_cp_loss += compactness_loss.data.item()
        tr_sp_loss += separateness_loss.data.item()
        tr_tot_loss += loss.data.item()

        progress_bar.set_postfix({'loss': f"{tr_tot_loss:.4f}"})
        
    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']
    train_writer.add_scalar('learning_rate', current_lr, epoch)
    px_loss = tr_px_loss/len(train_batch)
    cp_loss = tr_cp_loss/len(train_batch)
    sp_loss = tr_sp_loss/len(train_batch)
    tot_loss = tr_tot_loss/len(train_batch)
    train_writer.add_scalar(f"model/train-{args.method}-pixel-loss", px_loss, epoch)
    train_writer.add_scalar(f"model/train-{args.method}-compactness-loss", cp_loss, epoch)
    train_writer.add_scalar(f"model/train-{args.method}-separateness-loss", sp_loss, epoch)
    train_writer.add_scalar(f"model/train-{args.method}-total-loss", tot_loss, epoch)

    # TODO print memory items
    print(f"epoch {epoch:3d}/{args.epochs:3d}: total_loss: {tot_loss:.4f} | pixel_loss: {px_loss:.4f} | compactness_loss: {cp_loss:.4f} | separateness_loss: {sp_loss:.4f}")

    # validation
    if epoch == 1 or epoch % args.val_epoch == 0:
        model.eval()
        val_px_loss, val_cp_loss, val_sp_loss, val_tot_loss = 0.0, 0.0, 0.0, 0.0
        progress_bar = tqdm(val_batch)

        for j,(imgs,_) in enumerate(progress_bar):
            progress_bar.set_description(f"Validation Epoch {epoch:3d}/{args.epochs:3d}")

            imgs = Variable(imgs).cuda()
            
            if args.method == 'pred':
                outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs[:,0:img_slice], m_items, True)
            else:
                outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs, m_items, True)
            
            optimizer.zero_grad()
            if args.method == 'pred':
                loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:,img_slice:]))
            else:
                loss_pixel = torch.mean(loss_func_mse(outputs, imgs))
                
            val_loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss

            val_px_loss += loss_pixel.data.item()
            val_cp_loss += compactness_loss.data.item()
            val_sp_loss += separateness_loss.data.item()
            val_tot_loss += val_loss.data.item()

            progress_bar.set_postfix({'loss': f"{val_tot_loss:.4f}"})
            
        px_loss = val_px_loss/len(val_batch)
        cp_loss = val_cp_loss/len(val_batch)
        sp_loss = val_sp_loss/len(val_batch)
        tot_loss = val_tot_loss/len(val_batch)
        train_writer.add_scalar(f"model/val-{args.method}-pixel-loss", px_loss, epoch)
        train_writer.add_scalar(f"model/val-{args.method}-compactness-loss", cp_loss, epoch)
        train_writer.add_scalar(f"model/val-{args.method}-separateness-loss", sp_loss, epoch)
        train_writer.add_scalar(f"model/val-{args.method}-total-loss", tot_loss, epoch)

        print(f"Validation epoch {epoch:3d}/{args.epochs:3d}: total_loss: {tot_loss:.4f} | pixel_loss: {px_loss:.4f} | compactness_loss: {cp_loss:.4f} | separateness_loss: {sp_loss:.4f}")

    # save model
    if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:   
        torch.save(model, os.path.join(log_dir, f'{epoch:03d}_model.pth'))
        torch.save(m_items, os.path.join(log_dir, f'{epoch:03d}_keys.pt'))
    
print(f'Training finished in {(time.time() - time_start)/60/60:.2f} hours')

sys.stdout = orig_stdout
f.close()

# TODO: run on training test
