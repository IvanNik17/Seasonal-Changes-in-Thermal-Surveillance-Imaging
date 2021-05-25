from argparse import ArgumentParser
import os, sys
import cv2
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from torchsummary import summary
import torch.nn.functional as F

sys.path.append('../../loaders/pytorch_lightning/')
from datamodule import DataModule

from models.autoencoder import Autoencoder

torch.backends.cudnn.benchmark = True

def train(hparams, dm):
    #logger = loggers.TensorBoardLogger(hparams.log_dir, name=f"da{hparams.data_root}_is{hparams.image_size}_nc{hparams.nc}")
    model = Autoencoder(hparams)
    # print detailed summary with estimated network size
    #summary(model, (hparams.nc, hparams.image_width, hparams.image_height), device="cpu")
    trainer = Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs)
    trainer.fit(model, dm)
    trainer.test(model)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(model.encoder, os.path.join(output_dir,"encoder.pt"))
    torch.save(model.decoder, os.path.join(output_dir,"decoder.pt"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_list", type=str, default="output/train_total.txt", help="list of training images")
    parser.add_argument("--val_list", type=str, default="output/val_all.txt", help="list of validation images")
    #parser.add_argument("--image_dir", type=str, default="data/train/", help="View root directory")
    #parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=12, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--image_width", type=int, default=384, help="Width of images")
    parser.add_argument("--image_height", type=int, default=288, help="Height of images")
    parser.add_argument("--max_epochs", type=int, default=100, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size during training")
    parser.add_argument("--nc", type=int, default=1, help="Number of channels in the training images")
    parser.add_argument("--norm", type=int, default=0, help="Normalize or not")
    parser.add_argument("--nz", type=int, default=8, help="Size of latent vector z")
    parser.add_argument("--nfe", type=int, default=32, help="Size of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=32, help="Size of feature maps in decoder")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs. Use 0 for CPU mode")

    args = parser.parse_args()

    images_path = "/home/aau/github/data/thermal/sensor_paper"

    metadata_train = pd.read_csv("../../splits/coldest_day.csv")
    metadata_train["DateTime"] = pd.to_datetime(metadata_train['DateTime'])

    metadata_test = pd.read_csv("../../splits/hottest_day.csv")
    metadata_test["DateTime"] = pd.to_datetime(metadata_train['DateTime'])

    cfg = {
       'img_dir': images_path,
       'metadata_train': metadata_train,
       'metadata_test': metadata_test,
       'get_metadata': False,
       'batch_size': 64,
    }

    # instantiate the class and give it the cfg dictionary, call the setup
    dm = DataModule(cfg)
    dm.setup()

    print("sample shape {}".format(dm.data_val[0].shape))

    train(args, dm)
