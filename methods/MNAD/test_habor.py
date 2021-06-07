# python test_habor.py --method recon --t_length 1 --model_dir=/home/jacob/data/models/mnad/habor/recon/one_day_001
# python test_habor.py --method pred --t_length 5 --c 1 --model_dir=/home/jacob/data/models/mnad/habor/pred/one_day_001
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
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model.Reconstruction import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob
from tqdm.autonotebook import tqdm  # TODO autonotebook?
import pandas as pd
from shutil import copyfile

import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import json


def parse_args():
    parser = argparse.ArgumentParser(description="MNAD")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=1, help='channel of input images')
    parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
    parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
    parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
    parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
    parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    # parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
    parser.add_argument('--model_dir', type=str, help='directory of model')
    # parser.add_argument('--m_items_dir', type=str, help='directory of model')
    parser.add_argument('--checkpoint', type=int, default=100, help='directory of model')

    args = parser.parse_args()
    return args

def evaluate(args, test_folder, if_mask_mse=True, save_images=False):
    csv_name, extension = os.path.splitext(os.path.basename(test_folder))

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

    args_file = os.path.join(args.model_dir, 'train_args.json')
    with open(args_file, 'r') as fp:
        args_train = json.load(fp)
        dataset_type = args_train['dataset_type']

    # Loading dataset
    test_dataset = DataLoader('test', test_folder, transforms.Compose([
                 transforms.ToTensor(),            
                 ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                                 shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    loss_func_mse = nn.MSELoss(reduction='none')

    # Loading the trained model
    model_path = os.path.join(args.model_dir, f'{args.checkpoint:03d}_model.pth')
    model = torch.load(model_path)
    model.cuda()
    model.eval()

    memory_path = os.path.join(args.model_dir, f'{args.checkpoint:03d}_keys.pt')
    m_items = torch.load(memory_path)
    m_items_test = m_items.clone()

    if args.method == 'pred':
        img_slice = (args.t_length - 1) * args.c

    df = test_dataset.samples
    recon_error_key = 'MSE'
    df[recon_error_key] = ""

    if if_mask_mse:
        mask_path = r'/home/jacob/code/Seasonal-Changes-in-Thermal-Surveillance-Imaging/pre_processing/mask_ropes_water.png'
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        mse_masked_key = 'MSE_moving_bkgrnd'
        df[mse_masked_key] = ""

    progress_bar = tqdm(test_batch)
    for k, (imgs, img_path) in enumerate(progress_bar):
        imgs = Variable(imgs).cuda()

        if args.method == 'pred':
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:img_slice], m_items_test, False)
            inp = (imgs[0, img_slice:] + 1) / 2
            rec = (outputs[0] + 1) / 2
            err = loss_func_mse(inp, rec)
            mse_imgs = torch.mean(err).item()
            mse_feas = compactness_loss.item()

            # Calculating the threshold for updating at the test time
            point_sc = point_score(outputs, imgs[:,img_slice:])
        
        else:
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items_test, False)
            inp = (imgs[0] + 1) / 2
            rec = (outputs[0] + 1) / 2
            err = loss_func_mse(inp, rec)
            mse_imgs = torch.mean(err).item()
            mse_feas = compactness_loss.item()

            # Calculating the threshold for updating at the test time
            point_sc = point_score(outputs, imgs)

        if  point_sc < args.th:
            query = F.normalize(feas, dim=1)
            query = query.permute(0,2,3,1) # b X h X w X d
            m_items_test = model.memory.update(query, m_items_test, False)

        if if_mask_mse:
            inp_img = inp.detach().cpu().numpy()[0]
            rec_img = rec.detach().cpu().numpy()[0]
            mse_masked = compute_masked_mse(inp_img, rec_img, mask)
            df.iloc[k, df.columns.get_loc(mse_masked_key)] = mse_masked

        if save_images:
            inp_img = (inp.detach().cpu().numpy()[0] * 255).astype(np.uint8)
            rec_img = (rec.detach().cpu().numpy()[0] * 255).astype(np.uint8)
            err_img = (err.detach().cpu().numpy()[0] * 255).astype(np.uint8)

            inp_img_rgb = cv2.cvtColor(inp_img, cv2.COLOR_GRAY2RGB)
            rec_img_rgb = cv2.cvtColor(rec_img, cv2.COLOR_GRAY2RGB)
            # err_img_rgb2 = cv2.cvtColor(err_img, cv2.COLOR_GRAY2RGB)
            err_img_rgb = cv2.applyColorMap(err_img, cv2.COLORMAP_JET)
            combined_img = np.concatenate((inp_img_rgb, rec_img_rgb, err_img_rgb), axis=1)

            save_dir = os.path.join(args.model_dir, 'rec_errors_' + csv_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            split = img_path[0].split('/')
            datetime, clip, img_name = split[-3:]
            img_save_name = f'{mse_imgs:.4f}_{datetime}_{clip}_{img_name}'
            img_save_path = os.path.join(save_dir, img_save_name)
            cv2.imwrite(img_save_path, combined_img)

        df.iloc[k, df.columns.get_loc(recon_error_key)] = mse_imgs

    # Save scores
    # df_save_path_filename = csv_name + f'_mnad_{args.method}_{dataset_type}' + extension
    df_save_path_filename = 'results_' + csv_name + extension
    df_save_path = os.path.join(args.model_dir, df_save_path_filename)
    df.to_csv(df_save_path, index=False)


def compute_masked_mse(img_input, img_rec, mask):
    mask = cv2.bitwise_not(mask)
    img_masked = cv2.bitwise_and(img_input, img_input, mask=mask)
    rec_masked = cv2.bitwise_and(img_rec, img_rec, mask=mask)

    if img_masked.max() > 1:
        img_masked = img_masked / 255.0
    if rec_masked.max() > 1:
        rec_masked = rec_masked / 255.0

    # mse = (np.square(img_input - img_rec )).mean()
    mse_masked = (np.square(img_masked - rec_masked)).mean()
    return mse_masked


def topX_errors(csv_path, dataset_path, model_dir, topX=100):
    recon_error_key = 'MSE'
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=[recon_error_key], ascending=False)
    df = df[0:topX]
    assert len(df) == topX

    test_name = os.path.splitext(os.path.basename(csv_path))[0]
    top_errors_name = 'top_errors_' + test_name
    save_dir_path = os.path.join(model_dir, top_errors_name)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    top_errors_csv = os.path.join(save_dir_path, top_errors_name + '.csv')
    df.to_csv(top_errors_csv, index_label='Index')  # TODO Test index column

    # save images
    for index, sample in df.iterrows():
        mse = sample[recon_error_key]
        image_path_src = os.path.join(dataset_path,
                                      str(sample['Folder name']),
                                      sample['Clip Name'],
                                      f"image_{int(sample['Image Number']):04}.jpg")
        # copy image
        image_filename = f"{mse:0.4f}_{sample['Folder name']}_{sample['Clip Name']}_{sample['Image Number']}.jpg"
        image_path_dst = os.path.join(save_dir_path, image_filename)
        copyfile(image_path_src, image_path_dst)


def plot_reconstruction_error(csv_path, save_img_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['Folder name', 'Clip Name', 'Image Number'], ascending=True)
    #  TODO use grouping
    # grouped = df.groupby(["Cluster", "Week"]).agg({'Overload': ['min', 'mean', 'max']}).unstack("Cluster")
    df['folder_clip_number'] = df['Folder name'].astype(str) + '_' + df['Clip Name'] + '_' + df['Image Number']

    ax = sns.lineplot(data=df, x="folder_clip_number", y="recon_error")
    ax.set(xticklabels=[])
    figure = ax.get_figure()
    save_figure(figure, save_img_path)


def plot_reconstruction_error_clip(csv_path, save_img_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['Folder name', 'Clip Name', 'Image Number'], ascending=True)
    df['folder_clip'] = df['Folder name'].astype(str) + '_' + df['Clip Name']

    ax = sns.lineplot(data=df, x="folder_clip", y="recon_error")
    plt.xticks(rotation=-90)
    figure = ax.get_figure()
    save_figure(figure, save_img_path)


def plot_reconstruction_error_folder(csv_path, save_img_path):
    df = pd.read_csv(csv_path)

    ax = sns.lineplot(data=df, x="Folder name", y="recon_error")
    plt.xticks(rotation=-90)
    figure = ax.get_figure()
    save_figure(figure, save_img_path)


def save_figure(figure, destination_path):
    destination_dirname = os.path.dirname(destination_path)
    if not os.path.exists(destination_dirname):
        os.makedirs(destination_dirname)

    figure_file_path = destination_path
    figure.savefig(figure_file_path, bbox_inches='tight', pad_inches=0.1, dpi=100)


def compute_mean_error(method_dir):
    models = ['feb_day', 'feb_week', 'feb_month']
    months = ['results_jan_month', 'results_apr_month', 'results_aug_month']

    for model in models:
        for month in months:
            path = os.path.join(method_dir, model, month + '.csv')
            df = pd.read_csv(path)

            mean_error = df['MSE'].mean()
            print(f"{model:10s} {month} mean error: {mean_error:.3f}")


if __name__ == "__main__":
    args = parse_args()

    evaluate(args, '/home/jacob/code/Seasonal-Changes-in-Thermal-Surveillance-Imaging/splits/jan_month.csv', True, False)
    evaluate(args, '/home/jacob/code/Seasonal-Changes-in-Thermal-Surveillance-Imaging/splits/apr_month.csv', True, False)
    evaluate(args, '/home/jacob/code/Seasonal-Changes-in-Thermal-Surveillance-Imaging/splits/aug_month.csv', True, False)

    # compute_mean_error(f'/home/jacob/data/models/mnad/habor/{args.method}')

    # # top errors
    # dataset_path = r'/home/jacob/data/habor/image_dataset/'
    # csv_path = os.path.join(args.model_dir, 'results_jan_month.csv')
    # topX_errors(csv_path, dataset_path, args.model_dir, topX=100)
    # csv_path = os.path.join(args.model_dir, 'results_apr_month.csv')
    # topX_errors(csv_path, dataset_path, args.model_dir, topX=100)
    # csv_path = os.path.join(args.model_dir, 'results_aug_month.csv')
    # topX_errors(csv_path, dataset_path, args.model_dir, topX=100)

    # save_img_path = os.path.join(model_dir, 'img', 'val_reconstruction_error.png')
    # plot_reconstruction_error(csv_path, save_img_path)
    #
    # save_img_path = os.path.join(model_dir, 'img', 'val_reconstruction_error_clip.png')
    # plot_reconstruction_error_clip(csv_path, save_img_path)
    #
    # save_img_path = os.path.join(model_dir, 'img', 'val_reconstruction_error_folder.png')
    # plot_reconstruction_error_folder(csv_path, save_img_path)

