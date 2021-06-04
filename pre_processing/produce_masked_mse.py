# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:25:18 2021

@author: IvanTower
"""
import numpy as np
import cv2


def masked_mse(img_input, img_rec, mask):
    mask = cv2.bitwise_not(mask)
    
    img_masked = cv2.bitwise_and(img_input, img_input, mask=mask)
    rec_masked = cv2.bitwise_and(img_rec, img_rec, mask=mask)
    
    if img_masked.max() > 1:
        img_masked = img_masked / 255.0
        
    if rec_masked.max() > 1:
        rec_masked = rec_masked / 255.0
    
    mse_masked = (np.square(img_masked - rec_masked)).mean()
    
    return mse_masked
