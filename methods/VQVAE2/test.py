import argparse
import os, sys
import cv2
import numpy as np
import csv

import torch
import pytorch_lightning as pl
#from torchsummary import summary
import torch.nn.functional as F
from torchvision import transforms

from vqvae2 import VQVAE2

from config import hparams
from glob import glob

from thermal_dataset import ThermalDataset

import albumentations as Augment
import torchvision.utils as vutils

if hparams.in_channels == 3:
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
else:
    mean, std = [0.5], [0.5]

def normalize(img):
    if hparams.in_channels == 3:
        img[0] = (img[0] - 0.5) / 0.5
        img[1] = (img[1] - 0.5) / 0.5
        img[2] = (img[2] - 0.5) / 0.5
    else:
        img[0] = (img[0] - 0.5) / 0.5
    return img

test_t = Augment.Compose([Augment.SmallestMaxSize(max_size=hparams.image_height, interpolation=cv2.INTER_LINEAR, always_apply=True),
                          Augment.CenterCrop(hparams.image_height, hparams.image_height, always_apply=True),
                          Augment.Normalize(mean, std)
              ])


data_val = ThermalDataset(os.path.join(hparams.data_path,'val'),
                             transforms=test_t,
                             n_channels=hparams.in_channels)

def process_video(video_path, net):

    cap = cv2.VideoCapture(video_path)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        key = predict(img, net)
        if key == 27:
            break

def process_img(img_path, net):

    if os.path.isfile(img_path):
        print(img_path)
        output_path = os.path.join('output','x_'+os.path.basename(img_path))

        #if hparams.in_channels == 1:
        #    img = cv2.imread(img_path, -1)
        #    img = img[:, :, np.newaxis]
        #else:
        img = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        key = predict(img, net)


def process_img_dir(img_dir, net):
    img_list = sorted([y for y in glob(os.path.join(img_dir, 'val/*/*/*.jpg'))])
    if len(img_list):
        print("Found {} files".format(len(img_list)))
    else:
        print("did not find any files")

    for img_path in img_list:
        if os.path.isfile(img_path):
            print(img_path)
            output_path = os.path.join('output','x_'+os.path.basename(img_path))

            if hparams.in_channels == 1:
                img = cv2.imread(img_path, -1)
                img = img[:, :, np.newaxis]
            else:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            key = predict(img, net)
            if key == 27:
                break

def predict(img, net):
    if hparams.in_channels == 1:
        img = img[:, :, np.newaxis]
    x = img.transpose((2, 0, 1))/255.0
    x = normalize(x)
    x = torch.as_tensor(x, dtype=torch.float32)

    rec, diffs, encs, recs = net(x.unsqueeze(0))
    rec = rec[0]

    #
    diff = x - rec
    diff = torch.abs(diff)
    diff = torch.clamp(diff, min=0.0, max=1.0)
    diff = diff.mul(255).permute(1, 2, 0).byte().numpy()
    #cv2.imwrite(output_path.replace('x_','diff'),diff)
    diff = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    cv2.imshow("diff",diff)

    rec = rec * 0.5 + 0.5
    rec = torch.clamp(rec, min=0.0, max=1.0)
    rec = rec.mul(255).permute(1, 2, 0).byte().numpy()

    cv2.imshow("rec",rec)

    x = x * 0.5 + 0.5
    input = x.mul(255).permute(1, 2, 0).byte().numpy()

    cv2.imshow("input",input)

    img_stack = cv2.vconcat([cv2.cvtColor(input, cv2.COLOR_GRAY2BGR), cv2.cvtColor(rec, cv2.COLOR_GRAY2BGR), diff])
    #cv2.imwrite(output_path.replace('x_','input_rec_diff'),img_stack)

    #cv2.imshow("test",img)
    return cv2.waitKey()



if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/model.pt', help="path to the checkpoint file")
    args = vars(ap.parse_args())

    data_dir = '/home/aau/github/thermal_autoencoder/data/feb21/'

    with torch.no_grad():

        net = VQVAE2(in_channels=hparams.in_channels,
                     hidden_channels=hparams.hidden_channels,
                     embed_dim=hparams.embed_dim,
                     nb_entries=hparams.nb_entries,
                     nb_levels=hparams.nb_levels,
                     scaling_rates=hparams.scaling_rates)

        net.load_state_dict(torch.load(args['checkpoint']))
        #net = torch.load(args['checkpoint'])
        net.eval()

        process_img("elephant.jpg", net)
        #process_video("03-03-2016 10_00_28 (UTC+01_00).mkv", net)
        #process_img_dir(img_dir, net)
