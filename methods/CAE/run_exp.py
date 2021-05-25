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

from argparse import Namespace

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp", type=str, default="month", help="specify which experiment to run: day, week, month")
    parser.add_argument("--path", type=str, default="/home/aau/github/data/thermal/sensor_paper", help="specify where images are located")
    parser.add_argument("--model", type=str, default="CAE", help="specify which experiment model to use")
    parser.add_argument("--train", default=False, action="store_true", help="Set if necessary to train a new model")
    args = parser.parse_args()

    # defualt configuration
    hparams = Namespace(**{'model': 'VQVAE2',
                           'dataset': 'day',
                           'season': 'coldest',
                           'img_dir': '/home/aau/github/data/thermal/sensor_paper',
                           'train_selection': None,
                           'test_selection': None,
                           #'image_width': 384,
                           #'image_height': 288,
                           'get_metadata': False,
                           # model
                           'nc': 1,
                           'nz': 8,
                           'nfe': 32,
                           'nfd': 32,
                           # training
                           'log_dir': 'lightning_logs',
                           'gpus': 1,
                           'max_epochs': 100,
                           'learning_rate': 1e-4,
                           'batch_size': 128,
                           'num_workers':12})

    hparams.model = args.model

    images_path = "/home/aau/github/data/thermal/sensor_paper"

    logger = loggers.TensorBoardLogger(hparams.log_dir, name=f"{hparams.season}_{hparams.dataset}", default_hp_metric=False)

    model = Autoencoder(hparams)

    if args.train:
        train = pd.read_csv("../../splits/{}_{}.csv".format(hparams.season, hparams.dataset))
        train["DateTime"] = pd.to_datetime(train['DateTime'])
        hparams.train_selection = train

        test = pd.read_csv("../../splits/median_month.csv".format(hparams.dataset))
        test["DateTime"] = pd.to_datetime(test['DateTime'])
        hparams.test_selection = test

        dm = DataModule(hparams)
        dm.setup()

        print("sample shape {}".format(dm.data_val[0].shape))

        # print detailed summary with estimated network size
        #summary(model, (hparams.nc, hparams.image_width, hparams.image_height), device="cpu")
        trainer = Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs, logger=logger)
        trainer.fit(model, dm)

        torch.save(model.encoder, "trained_models/{}_{}_encoder.pt".format(hparams.season, hparams.dataset))
        torch.save(model.decoder, "trained_models/{}_{}_decoder.pt".format(hparams.season, hparams.dataset))

        trainer.test(model)

    model.encoder = torch.load("trained_models/{}_{}_encoder.pt".format(hparams.season, hparams.dataset))
    model.decoder = torch.load("trained_models/{}_{}_decoder.pt".format(hparams.season, hparams.dataset))

    #print(model)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    #from torchsummary import summary
    #summary(model, (1, 64, 192))

    model.encoder.eval()
    model.decoder.eval()

    hparams.get_metadata = True

    test = pd.read_csv("../../splits/hottest_month.csv".format(hparams.dataset))
    test["DateTime"] = pd.to_datetime(test['DateTime'])
    hparams.test_selection = test

    dm = DataModule(hparams)
    dm.setup()

    for batch_id, batch in enumerate(dm.test_dataloader()):
        imgs, paths, metas = batch
        encs = model.encoder(imgs)
        recs = model.decoder(encs)

        for frame, path, meta, enc, rec in zip(imgs, paths, metas, encs, recs):
            print(meta)
            rec = rec[0].mul(255).byte().numpy()

            cv2.imshow("rec",rec)
            key = cv2.waitKey()
            if key == 27:
                break
        break
