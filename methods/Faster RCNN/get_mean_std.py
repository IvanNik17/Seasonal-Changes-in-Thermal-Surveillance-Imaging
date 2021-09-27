
import cv2
import numpy as np
import os
import glob

data = 'data/harborfront/train/2235/images/'
img_type = '.jpg'

files = glob.glob(data+'/*'+img_type)

mean = []
std = []
for f in files:
    im = cv2.imread(f, 0)
    mean.append(np.mean(im))
    std.append(np.std(im))

print(np.sum(mean)/len(mean))
print(np.sum(std)/len(std))
    