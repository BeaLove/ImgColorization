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
import misc.npy_loader.loader as npy

import data_loader as dl

import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-2, type=float,
                    help='learning rate')  # TODO test initial lr of 1e-2 w cosine annealing
parser.add_argument('--betas', default=(0.9, 0.999), help='betas for ADAM')
parser.add_argument('--loss', default='RarityWeighted', help='loss function')
opt = parser.parse_args()

"""
 Code found from: https://github.com/richzhang/colorization
"""

weight_mix = npy.load('weight_distribution_mix_with_uniform_distribution_normalized')


class Colorization_model_Alex(pl.LightningModule):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_bins=441, loss=opt.loss):
        super(Colorization_model_Alex, self).__init__()
        if loss == 'RarityWeighted':
            self.data_loaders = dl.return_loaders(soft_encoding=True)
            self.loss_criterion = RarityWeightedLoss(weight_mix)
        elif loss == 'L2':
            self.data_loaders = dl.return_loaders(soft_encoding=False)
            self.loss_criterion = L2Loss()
        model1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True), ]
        model1 += [nn.ReLU(True), ]
        model1 += [norm_layer(64), ]

        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), ]
        model2 += [nn.ReLU(True), ]
        model2 += [norm_layer(128), ]

        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [norm_layer(256), ]

        model4 = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]

        model4 += [nn.Conv2d(256, num_bins, kernel_size=1, stride=1, padding=0, bias=True), ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)

        self.softmax = nn.LogSoftmax(dim=1)
        if loss == 'RarityWeighted':
            self.model5 = nn.Upsample(scale_factor=4, mode='bilinear')
        elif loss == 'L2':
            self.model5 = nn.Sequential(
                nn.Conv2d(num_bins, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False),
                nn.Upsample(scale_factor=4, mode='bilinear'))

    # self.model_out = nn.Conv2d(num_bins, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
    # self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, X):
        conv1_2 = self.model1(X)
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)

        '''try returning num bins to loss function'''
        #upsampled = self.model5(conv4_3)
        out_reg = self.softmax(conv4_3)
        # out_reg = self.upsample4(self.softmax(conv8_3))
        # out = self.upsample4(out_reg)
        return out_reg

    def training_step(self, batch, batch_idx):
        X, y = batch
        output = self.forward(X)
        loss = self.loss_criterion(output, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        output = self.forward(X)
        loss = self.loss_criterion(output, y)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, betas=opt.betas, weight_decay=1e-5)
        # T_max should be number of cycles to vary the learning rate, i set to 3 (12,000 steps if batch size is 25)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=1e-6,
                                                               T_max=12000)  # TODO comment out if you don't want to mess with
        return [optimizer], [scheduler]

    def predict_step(self, batch: int, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def on_epoch_end(self):
        global_step = self.global_step
        for name, param in self.named_parameters():
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
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='max'
    )
    '''log learning rate'''
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    model = Colorization_model_Alex(loss=opt.loss)  # TODO set loss as RarityWeighted or L2, default: L2
    logger = loggers.TensorBoardLogger(save_dir='logs/')
    print("using GPU", torch.cuda.is_available())
    trainer = Trainer(max_epochs=300,
                      #gpus=1,
                      logger=logger,  # use default tensorboard
                      log_every_n_steps=20,  # log every update step for debugging
                      limit_train_batches=1.0,
                      limit_val_batches=1.0,
                      check_val_every_n_epoch=5,
                      callbacks=[early_stop_call_back, lr_callback])
    trainer.fit(model)
    '''we may not need the below. lightning model can be loaded from last checkpoint'''
    os.makedirs('trained_models', exist_ok=True)
    name = 'ColorizationModelOverfitTest.pth'
    torch.save(model, os.path.join('trained_models', name))


if __name__ == '__main__':
    # start trainer
    run_trainer()
