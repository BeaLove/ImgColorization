# import libraries
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import EarlyStopping
from multiprocessing import Process
from loss import RarityWeightedLoss, L2Loss
#import misc.npy_loader.loader as npy
import numpy as np

import data_loader as dl

import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=3e-5, type=float,
                    help='learning rate')  # TODO test initial lr of 1e-2 w cosine annealing
parser.add_argument('--betas', default=(0.9, 0.999), help='betas for ADAM')
parser.add_argument('--loss', default='L2', help='loss function')
opt = parser.parse_args()

"""
 Code found from: https://github.com/richzhang/colorization
"""

weight_mix = np.load('../npy/weight_distribution_mix_with_uniform_distribution_reduced_normalized.npy')


class Colorization_model_Reduced(pl.LightningModule):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_bins=len(weight_mix), loss=opt.loss, batch_size=128, T_max=39000):
        super(Colorization_model_Reduced, self).__init__()
        self.T_max = T_max
        if loss == 'RarityWeighted':
            self.data_loaders = dl.return_loaders(batch_size=batch_size, soft_encoding=True)
            self.loss_criterion = RarityWeightedLoss(weight_mix)
        elif loss == 'L2':
            self.data_loaders = dl.return_loaders(batch_size=batch_size, soft_encoding=False)
            self.loss_criterion = torch.nn.MSELoss()
        model1 = [nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True), ]
        model1 += [nn.ReLU(True), ]
        model1 += [nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=True), ]
        model1 += [nn.ReLU(True), ]
        model1 += [norm_layer(16), ]

        model2 = [nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True), ]
        model2 += [nn.ReLU(True), ]
        model2 += [nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True), ]
        model2 += [nn.ReLU(True), ]
        model2 += [norm_layer(32), ]

        model3 = [nn.Conv2d(32, 64, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(64, 128, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [norm_layer(128), ]

        model4 = nn.Conv2d(128, num_bins, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)

        '''model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [norm_layer(512), ]'''

        '''model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [norm_layer(512), ]'''

        '''model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [norm_layer(512), ]'''

        '''model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [norm_layer(512), ]'''

        '''model8 = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]

        model8 += [nn.Conv2d(256, num_bins, kernel_size=1, stride=1, padding=0, bias=True), ]'''

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = model4


        if loss == 'RarityWeighted':
            self.out_layer = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear'), nn.LogSoftmax(dim=1))

        elif loss == 'L2':
            self.out_layer = nn.Sequential(
                nn.Conv2d(num_bins, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False),
                nn.Upsample(scale_factor=4, mode='bilinear'))

    def forward(self, X):
        conv1_2 = self.model1(X)
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4 = self.model4(conv3_3)

        out = self.out_layer(conv4)

        '''try returning num bins to loss function'''
        #upsampled = self.model5(conv4_3)

        return out

    def training_step(self, batch, batch_idx):
        X, y = batch
        output = self.forward(X)
        loss = self.loss_criterion(output, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        output = self.forward(X)
        loss = self.loss_criterion(output, y)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        output = self.forward(X)
        loss = self.loss_criterion(output, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, betas=opt.betas, weight_decay=1e-5)
        # T_max should be number of cycles to vary the learning rate, computed by batch size and epoch before initialization
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=1e-7,
                                                               T_max=self.T_max)  # TODO comment out if you don't want to mess with
        return [optimizer], [scheduler]

    def predict_step(self, batch: int, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def on_train_epoch_end(self):
        global_step = self.global_ste
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name + " grad", param.grad, global_step)
            self.logger.experiment.add_histogram(name, param, global_step)


    # @pl.data_loader
    def train_dataloader(self):
        return self.data_loaders['train']

    # @pl.data_loader
    def test_dataloader(self):
        return self.data_loaders['test']

    def val_dataloader(self):
        return self.data_loaders['validation']


# def configure_optimizers(self):
# Dummy optimizer for testing
#	return torch.optim.Adam(self.parameters(), lr=1e-5)


def run_trainer():
    early_stop_call_back = EarlyStopping(
        monitor='val_loss_epoch',
        min_delta=0.00,
        check_finite=True,
        patience=15,
        verbose=True,
        mode='min'
    )
    '''log learning rate'''
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=3,
        monitor='val_loss_epoch',
        save_last=True,
        verbose=True,
        every_n_val_epochs=1
    )

    max_epochs = 50
    batch_size=128
    T_max = np.floor(100000/batch_size)*max_epochs
    "/logs/default/version_87/checkpoints/epoch=2-step=2345.ckpt"
    model = Colorization_model_Reduced(loss=opt.loss, batch_size=batch_size, T_max=T_max)  # TODO set loss as RarityWeighted or L2, default: L2
    logger = loggers.TensorBoardLogger(save_dir='logs/')
    if torch.cuda.is_available():
        print("using GPU")
        num_gpus = torch.cuda.device_count()
    else:
        print("using CPU")
        num_gpus=0
    '''trainer = Trainer(max_epochs=max_epochs,
                      min_epochs=4,
                      gpus=num_gpus,
                      logger=logger,  # use default tensorboard
                      log_every_n_steps=20,  # log every update step for debugging
                      limit_train_batches=1.0,
                      limit_val_batches=0.7,
                      check_val_every_n_epoch=1,
                      callbacks=[lr_callback])'''
    trainer = Trainer(
                      max_epochs=max_epochs,
                      gpus=num_gpus,
                      logger=logger,  # use default tensorboard
                      log_every_n_steps=1,  # log every update step for debugging
                      limit_train_batches=1.0,
                      limit_val_batches=1.0,
                      check_val_every_n_epoch=1,
                      callbacks=[lr_callback, early_stop_call_back, checkpoint_callback]
                      )
    trainer.fit(model)


if __name__ == '__main__':
    #start trainer
    run_trainer()