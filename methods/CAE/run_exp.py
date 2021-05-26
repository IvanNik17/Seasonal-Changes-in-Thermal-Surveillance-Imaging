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

def train(model, hparams, logger):

    train = pd.read_csv("../../splits/{}_{}.csv".format(hparams.season, hparams.dataset))
    train["DateTime"] = pd.to_datetime(train['DateTime'])
    hparams.train_selection = train

    test = pd.read_csv("../../splits/median_month.csv")
    test["DateTime"] = pd.to_datetime(test['DateTime'])
    hparams.test_selection = test

    dm = DataModule(hparams)
    dm.setup()

    print("Training set contains {} samples".format(len(dm.data_train)))

    print("with shape {}".format(dm.data_train[0].shape))

    # print detailed summary with estimated network size
    #summary(model, (hparams.nc, hparams.image_width, hparams.image_height), device="cpu")
    trainer = Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs, logger=logger)
    trainer.fit(model, dm)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    exp_dir = os.path.join(output_dir,"{}_{}".format(hparams.season, hparams.dataset))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    torch.save(model.encoder, os.path.join(exp_dir,"encoder.pt"))
    torch.save(model.decoder, os.path.join(exp_dir,"decoder.pt"))

    trainer.test(model)

def test(model, hparams, test_sets, show=False):

    # runs on CPU
    with torch.no_grad():

        model.encoder = torch.load("trained_models/{}_{}/encoder.pt".format(hparams.season, hparams.dataset))
        model.decoder = torch.load("trained_models/{}_{}/decoder.pt".format(hparams.season, hparams.dataset))

        #print(model)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model = model.to(device)
        #from torchsummary import summary
        #summary(model, (1, 64, 192))

        model.encoder.eval()
        model.decoder.eval()

        hparams.get_metadata = True

        for test_set in test_sets:

            test = pd.read_csv("../../splits/{}.csv".format(test_set))
            test["DateTime"] = pd.to_datetime(test['DateTime'])
            hparams.test_selection = test

            dm = DataModule(hparams)
            dm.setup(stage='test')

            results_list = []

            for batch_id, batch in enumerate(dm.test_dataloader()):
                imgs, paths, metas = batch
                encs = model.encoder(imgs)
                recs = model.decoder(encs)

                for img, path, meta, enc, rec in zip(imgs, paths, metas, encs, recs):
                    #folder_name, clip_name, image_number, DateTime = meta.split(',')[:4]
                    results = meta.split(',')

                    loss = F.mse_loss(rec, img).numpy()

                    results_list.append(results+[str(np.round(loss, 4))])
                    if show:
                        diff = img - rec
                        diff = torch.abs(diff)
                        diff = diff[0].mul(255).byte().numpy()

                        img = img[0].mul(255).byte().numpy()
                        rec = rec[0].mul(255).byte().numpy()

                        cv2.imshow("in_vs_rec_vs_diff",cv2.vconcat([img, rec, diff]))
                        key = cv2.waitKey()
                        if key == 27:
                            break

                if show:
                    if key == 27:
                        break

            # save all the new info
            results_df = pd.DataFrame.from_records(results_list, columns=['Folder name',
                                                                          'Clip Name',
                                                                          'Image Number',
                                                                          'DateTime',
                                                                          'Temperature',
                                                                          'Humidity',
                                                                          'Precipitation',
                                                                          'Dew Point',
                                                                          'Wind Direction',
                                                                          'Wind Speed',
                                                                          'Sun Radiation Intensity',
                                                                          'Min of sunshine latest 10 min',
                                                                          'MSE'])

            results_df.to_csv("trained_models/{}_{}/results_{}.csv".format(hparams.season, hparams.dataset, test_set), index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp", type=str, default="month", help="specify which experiment to run: day, week, month")
    parser.add_argument("--path", type=str, default="/home/aau/github/data/thermal/sensor_paper", help="specify where images are located")
    parser.add_argument("--model", type=str, default="CAE", help="specify which experiment model to use")
    parser.add_argument("--train", default=False, action="store_true", help="Set if necessary to train a new model")
    parser.add_argument("--show", default=False, action="store_true", help="Set to show input and reconstructions")

    args = parser.parse_args()

    # defualt configuration
    hparams = Namespace(**{'model': 'VQVAE2',
                           'dataset': 'day',
                           'season': 'coldest',
                           'img_dir': '/home/aau/github/data/thermal/sensor_paper',
                           'train_selection': None,
                           'test_selection': None,
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
    hparams.dataset = args.exp

    logger = loggers.TensorBoardLogger(hparams.log_dir, name=f"{hparams.season}_{hparams.dataset}", default_hp_metric=False)

    model = Autoencoder(hparams)

    if args.train:
        train(model, hparams, logger)

    test_sets = ['best_case_month', 'median_month', 'hottest_month']

    test(model, hparams, test_sets, show=args.show)
