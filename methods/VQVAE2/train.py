import argparse
import os, sys
import cv2
import numpy as np
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
#from torchsummary import summary
import torch.nn.functional as F

from autoencoder import Autoencoder

from config import hparams

import albumentations as Augment
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

class MyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

        if 'thermal' in hparams.dataset:

            if hparams.in_channels == 1:
                mean, std = [0.5], [0.5]
            else:
                mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

            self.train_t = Augment.Compose([#Augment.ToGray(p=1.0),
                                          #Augment.SmallestMaxSize(max_size=hparams.image_height, interpolation=cv2.INTER_LINEAR, always_apply=True),
                                          #Augment.RandomCrop(hparams.image_height, hparams.image_height, always_apply=True),
                                          #Augment.HorizontalFlip(p=0.5),
                                          Augment.RandomBrightnessContrast(p=0.5),
                                          Augment.Normalize(mean, std)
                         ])

            self.test_t = Augment.Compose([
                                           #Augment.SmallestMaxSize(max_size=hparams.image_height, interpolation=cv2.INTER_LINEAR, always_apply=True),
                                           #Augment.CenterCrop(hparams.image_height, hparams.image_height, always_apply=True),
                                           Augment.Normalize(mean, std)
                          ])
        else:
            print("> Unknown dataset. Terminating")
            exit()

    def setup(self, stage=None):
        from thermal_dataset import ThermalDataset

        if stage == 'fit' or stage is None:
            self.data_train = ThermalDataset(os.path.join(hparams.data_path,'train'),
                                             transforms=self.train_t,
                                             n_channels=hparams.in_channels)
            self.data_val = ThermalDataset(os.path.join(hparams.data_path,'val'),
                                           transforms=self.test_t,
                                           n_channels=hparams.in_channels)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.n_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.n_workers, pin_memory=True)

    def n_training(self):
        return len(self.data_train)

def create_datamodule():
    #sys.path.append('../datasets')

    if hparams.dataset=='sewer':
        #from sewer.datamodule import SewerDataModule
        dm = MyDataModule()
        dm.setup()
        return dm
    elif hparams.dataset=='thermal':
        #from harbour_datamodule import HarbourDataModule
        dm = MyDataModule()
        dm.setup()
        return dm
    else:
        print("no such dataset: {}".format(hparams.dataset))
        return None

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

    dm = create_datamodule()
    if dm == None:
        print("failed to create datamodule")
        exit()

    #logger = loggers.TensorBoardLogger(config['logging_params']['save_dir'], name="{}_{}_{}".format(config['exp_params']['data'], config['exp_params']['image_size'],config['model_params']['in_channels']))
    model = Autoencoder(hparams)
    # print detailed summary with estimated network size
    #summary(model, (config['model_params']['in_channels'], config['exp_params']['image_size'], config['exp_params']['image_size']), device="cpu")
    trainer = Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs)

    trainer.fit(model, dm)
    #trainer.test(model)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    torch.save(model.net.state_dict(), os.path.join(output_dir,"model.pt"))

    #https://pytorch-lightning.readthedocs.io/en/latest/common/production_inference.html
