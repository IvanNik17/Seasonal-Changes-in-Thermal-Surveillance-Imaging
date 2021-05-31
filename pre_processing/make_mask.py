# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:22:00 2021

@author: IvanTower
"""

import numpy as np
import cv2
import os
import sys

import matplotlib.pyplot as plt

import pandas as pd

sys.path.append(r'../../loaders/pytorch_lightning/')

from dataset import Dataset


def produce_masked_images(cfg):
        
    
    dataset = Dataset(img_dir=cfg["image_path"], metadata = cfg["metadata"], return_metadata = True)
    
    mask = cv2.imread(cfg["mask_path"],0)
    
    for i, sample in enumerate(dataset):
        img, path, metadata = sample
        
        img_numpy = img.squeeze().detach().cpu().numpy()
        
        image_path_curr = '\\'.join(path.split('\\')[-3:-1])
        
        image_path_curr = os.path.join(cfg["main_save_path"], image_path_curr)
        
        masked = cv2.bitwise_and(img_numpy,img_numpy,mask = mask)
        
        
        if not os.path.exists(image_path_curr):

            os.makedirs(image_path_curr)
        
        cv2.imwrite(os.path.join(image_path_curr,path.split('\\')[-1]), masked*255)
        print(image_path_curr)


if __name__ == '__main__':
    
    mask_path = "mask.png"
    
    save_folder = r"masked"
    
    images_path = r"imageDir"
    metadata_file_name = r"image_metadata.csv"

    metadata_path = os.path.join(images_path,metadata_file_name)
    metadata = pd.read_csv(metadata_path)
    metadata["DateTime"] = pd.to_datetime(metadata['DateTime'], dayfirst = True)
    
    
    cfg = {
        'image_path': images_path,
        'metadata': metadata,
        'main_save_path': save_folder,
        'mask_path': mask_path,

    }
    
    produce_masked_images(cfg)
   