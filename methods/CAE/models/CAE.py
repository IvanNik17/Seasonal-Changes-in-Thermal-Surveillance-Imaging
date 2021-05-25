import torch
import torch.nn as nn

def create_encoder(hparams, kernel_size=4):
    return nn.Sequential(
        # input (nc) x 64 x 64
        nn.Conv2d(hparams.nc, hparams.nfe, kernel_size, 2, 1, bias=False),
        nn.BatchNorm2d(hparams.nfe),
        nn.LeakyReLU(True),
        # input (nfe) x 32 x 32
        nn.Conv2d(hparams.nfe, hparams.nfe * 2, kernel_size, 2, 1, bias=False),
        nn.BatchNorm2d(hparams.nfe * 2),
        nn.LeakyReLU(True),
        # input (nfe*2) x 16 x 16
        nn.Conv2d(hparams.nfe * 2, hparams.nfe * 4, kernel_size, 2, 1, bias=False),
        nn.BatchNorm2d(hparams.nfe * 4),
        nn.LeakyReLU(True),
        # input (nfe*4) x 8 x 8
        nn.Conv2d(hparams.nfe * 4, hparams.nfe * 8, kernel_size, 2, 1, bias=False),
        nn.BatchNorm2d(hparams.nfe * 8),
        nn.LeakyReLU(True),
        # input (nfe*8) x 4 x 4
        nn.Conv2d(hparams.nfe * 8, hparams.nz, kernel_size, 1, 0, bias=False),
        nn.BatchNorm2d(hparams.nz),
        nn.LeakyReLU(True)
        # output (nz) x 1 x 1
    )

def create_decoder(hparams, kernel_size=4):
    return nn.Sequential(
        # input (nz) x 1 x 1
        nn.ConvTranspose2d(hparams.nz, hparams.nfd * 8, kernel_size, 1, 0, bias=False),
        nn.BatchNorm2d(hparams.nfd * 8),
        nn.ReLU(True),
        # input (nfd*8) x 4 x 4
        nn.ConvTranspose2d(hparams.nfd * 8, hparams.nfd * 4, kernel_size, 2, 1, bias=False),
        nn.BatchNorm2d(hparams.nfd * 4),
        nn.ReLU(True),
        # input (nfd*4) x 8 x 8
        nn.ConvTranspose2d(hparams.nfd * 4, hparams.nfd * 2, kernel_size, 2, 1, bias=False),
        nn.BatchNorm2d(hparams.nfd * 2),
        nn.ReLU(True),
        # input (nfd*2) x 16 x 16
        nn.ConvTranspose2d(hparams.nfd * 2, hparams.nfd, kernel_size, 2, 1, bias=False),
        nn.BatchNorm2d(hparams.nfd),
        nn.ReLU(True),
        # input (nfd) x 32 x 32
        nn.ConvTranspose2d(hparams.nfd, hparams.nc, kernel_size, 2, 1, bias=False),
        nn.Sigmoid()
        # output (nc) x 64 x 64
    )
