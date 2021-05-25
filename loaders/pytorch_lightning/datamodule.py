# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:20:08 2021

@author: IvanTower
"""
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

import os
import pandas as pd
import albumentations as Augment

from src.dataset import Dataset

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.mean, self.std = [0.5], [0.5]

    def get_train_transform(self):
        return Augment.Compose([Augment.RandomBrightnessContrast(p=0.75),
                                Augment.Normalize(self.mean, self.std)
                               ])

    def get_test_transform(self):
        return Augment.Compose([Augment.Normalize(self.mean, self.std)
                               ])

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.data_train = Dataset(img_dir=self.cfg['img_dir'], metadata=self.cfg['metadata_train'], return_metadata = False, transforms=self.get_train_transform())

            val_start_idx = int(len(self.data_train) * 0.8)
            self.data_val = Subset(self.data_train, range(val_start_idx, len(self.data_train)))
            self.data_train = Subset(self.data_train, range(0, val_start_idx))

            self.data_test = Dataset(img_dir=self.cfg['img_dir'], metadata=self.cfg['metadata_test'], return_metadata = False, transforms=self.get_test_transform())

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.cfg['batch_size'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.cfg['batch_size'], shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.cfg['batch_size'], shuffle=False)


if __name__ == '__main__':

    #  path to the images, it should also contain the metadata csv file. The metadata file entries and images need to coincide
    images_path = "path-to-images"

    metadata_train = pd.read_csv("path-to/coldest_day.csv")
    metadata_train["DateTime"] = pd.to_datetime(metadata_train['DateTime'])

    metadata_test = pd.read_csv("path-to/hottest_day.csv")
    metadata_test["DateTime"] = pd.to_datetime(metadata_train['DateTime'])

    cfg = {
       'img_dir': images_path,
       'metadata_train': metadata_train,
       'metadata_test': metadata_test,
       'get_metadata': True,
       'batch_size': 16,
    }

    # instantiate the class and give it the cfg dictionary, call the setup
    dm = DataModule(cfg)
    dm.setup()

    #fixed_x = next(iter(dm.train_dataloader()))
    #print("train")
    #for sample in dm.data_train:
    #    img, path, meta = sample
    #    print(path)

    #fixed_x = next(iter(dm.val_dataloader()))

    #print("val")
    #for sample in dm.data_val:
    #    img, path, meta = sample
    #    print(path)
