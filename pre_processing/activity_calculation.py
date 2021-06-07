# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:22:18 2021

@author: IvanTower
"""

import numpy as np
import cv2
import os
import sys

import matplotlib.pyplot as plt

import pandas as pd


path = os.path.join(os.path.dirname(__file__), os.pardir)

if path not in sys.path:
    sys.path.append(path)
    
from loaders.pytorch_lightning.dataset import Dataset


def augment_activity(cfg, show_imgs=False, verbose=False):
    

    metadata = cfg['metadata'] 
    
    dataset = Dataset(img_dir=cfg["image_path"], selection = metadata, return_metadata = True)
    
    if cfg["mask_path"] != "":
        
        mask = cv2.imread(cfg["mask_path"],0)
    
    
    all_diff = []
    for i, sample in enumerate(dataset):
        img, path, meta = sample
        
        metadata_sep = meta.split(',')
        
        img_curr = img.squeeze().detach().cpu().numpy()
        
        img_curr=(img_curr/255).astype(np.float32)
        
        img_curr_num = int(metadata_sep[2])
        img_next_num = img_curr_num + 1
        img_next_name = "image_"+str(img_next_num).zfill(4)+".jpg"
        path_next = os.path.join(cfg["image_path"],metadata_sep[0],metadata_sep[1],img_next_name)
        
        if not os.path.exists(path_next):
            img_next_num = img_curr_num - 1
            img_next_name = "image_"+str(img_next_num).zfill(4)+".jpg"
            path_next = os.path.join(cfg["image_path"],metadata_sep[0],metadata_sep[1],img_next_name)
        
        
        img_next = cv2.imread(path_next,0)
        
        img_next=(img_next/255).astype(np.float32)
        
        if cfg["mask_path"] != "":
            img_curr = cv2.bitwise_and(img_curr,img_curr,mask = mask)
            img_next = cv2.bitwise_and(img_next,img_next,mask = mask)
        
        diff = cv2.absdiff(img_curr, img_next)
        
        all_diff.append(diff.mean())
        
        
        
        if verbose:
            print(f'i: {i:3d} DIFF {diff.mean()}')
        
        if show_imgs:
            show_img = cv2.vconcat([img_curr, img_next, diff])
        
            cv2.imshow("amount of difference",show_img)
            
            key = cv2.waitKey()
            if key == 27:
                break
       
    metadata["Activity"] =all_diff
        

    return metadata


if __name__ == '__main__':
    
    mask_path = "mask_ropes_water.png"
    
    
    images_path = r"E:\0_Monthly_videoPipeline\output_image_v3"
    
    metadata_path = r"D:\2021_workstuff\Seasonal-Changes-in-Thermal-Surveillance-Imaging\visualize_results\VQVAE\feb_day"
    metadata_name = r"results_apr.csv"
    
    
    metadata_path = os.path.join(metadata_path,metadata_name)
    metadata = pd.read_csv(metadata_path)
    metadata["DateTime"] = pd.to_datetime(metadata['DateTime'], dayfirst = True)


    
    
    
    cfg = {
        'image_path': images_path,
        'metadata': metadata,
        'mask_path': mask_path
    }
    
    metadata = augment_activity(cfg)
    
    
    
    # metadata_path = os.path.join(cfg["metadata_path"],cfg["metadata_name"])
    # metadata = pd.read_csv(metadata_path)
    # metadata["DateTime"] = pd.to_datetime(metadata['DateTime'], dayfirst = True)
    
    # dataset = Dataset(img_dir=cfg["image_path"], selection = metadata, return_metadata = True)
    
    
    # mask = cv2.imread(cfg["mask_path"],0)
    
    # use_mask = True
    
    # all_diff = []
    # for i, sample in enumerate(dataset):
    #     img, path, meta = sample
        
    #     metadata_sep = meta.split(',')
        
    #     img_curr = img.squeeze().detach().cpu().numpy()
        
    #     img_curr=(img_curr/255).astype(np.float32)
        
    #     img_curr_num = int(metadata_sep[2])
    #     img_next_num = img_curr_num + 1
    #     img_next_name = "image_"+str(img_next_num).zfill(4)+".jpg"
    #     path_next = os.path.join(images_path,metadata_sep[0],metadata_sep[1],img_next_name)
        
    #     if not os.path.exists(path_next):
    #         img_next_num = img_curr_num - 1
    #         img_next_name = "image_"+str(img_next_num).zfill(4)+".jpg"
    #         path_next = os.path.join(images_path,metadata_sep[0],metadata_sep[1],img_next_name)
            
        
        
    #     # path_list = list(path)
    #     # img_curr_num = int(path_list[-5])
    #     # path_next_list = path_list.copy()
    #     # path_next_list[-5] = str(img_curr_num+1)
    #     # path_next = ''.join(path_next_list)
        
    #     img_next = cv2.imread(path_next,0)
        
    #     img_next=(img_next/255).astype(np.float32)
        
    #     if use_mask:
    #         img_curr = cv2.bitwise_and(img_curr,img_curr,mask = mask)
    #         img_next = cv2.bitwise_and(img_next,img_next,mask = mask)
        
    #     diff = cv2.absdiff(img_curr, img_next)
        
    #     all_diff.append(diff.mean())
        
    #     show_img = cv2.vconcat([img_curr, img_next, diff])
        
    #     print(f'DIFF {diff.mean()}')
        
    #     # cv2.imshow("amount of difference",show_img)
        
    #     # key = cv2.waitKey()
    #     # if key == 27:
    #     #     break
        
    # metadata["Activity"] =all_diff
        
    # metadata.to_csv( os.path.join(save_folder,metadata_name) , index=False)    
    
