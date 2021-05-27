# import libraries


import os
from time import process_time
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
import torch
from torch._C import device
from torch.nn.modules.container import Sequential
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import EarlyStopping
from multiprocessing import Process

#import misc.npy_loader.loader as npy
import numpy as np
from PIL import Image
from skimage import io, color
import util_t as util

import model_eccv16
import model_siggraph

import data_loader as dl

import torch.nn as nn
import torch.nn.functional as nnf
import warnings

from pytorch_lightning.callbacks import ModelCheckpoint

warnings.filterwarnings('ignore')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=3e-4, type=float,
					help='learning rate')  # TODO test initial lr of 1e-2 w cosine annealing
parser.add_argument('--betas', default=(0.9, 0.999), help='betas for ADAM')
parser.add_argument('--loss', default='L2', help='loss function')
opt = parser.parse_args()

"""
 Code found from: https://github.com/richzhang/colorization
"""
device = torch.device('cuda:0')

points_in_hull = torch.tensor(np.load('pts_in_hull.npy')).to(device)
prior_probs = torch.tensor(np.load('prior_probs.npy')).to(device)

import sys

class Colorization_model_Reduced(pl.LightningModule):
	def __init__(self, sub_model = None, loss_weights = (1/4, 3/4), norm_layer=nn.BatchNorm2d, num_workers = 6, loss=opt.loss, batch_size=128, T_max=39000):
		super(Colorization_model_Reduced, self).__init__()
		self.T_max = T_max
		self.batch_size = batch_size
		self.loss_weights = loss_weights
		self.prior_probs = prior_probs
		self.points_in_hull = points_in_hull

		(self.sub_model_eccv, suspend) = sub_model
		#self.sub_model_sigge, = sub_model
		if suspend:
			self.sub_model_eccv.suspend_training() # don't update weights.



		self.data_loaders = dl.return_loaders(batch_size=batch_size, soft_encoding=False)
		# self.loss_criterion = L2Loss()
		
		# model1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True), ]
		# model1 += [nn.ReLU(True), ]
		# model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True), ]
		# model1 += [nn.ReLU(True), ]
		# model1 += [norm_layer(64), ]

		# model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), ]
		# model2 += [nn.ReLU(True), ]
		# model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True), ]
		# model2 += [nn.ReLU(True), ]
		# model2 += [norm_layer(128), ]

		# model3 = [nn.Conv2d(128, 256, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
		# model3 += [nn.ReLU(True), ]
		# model3 += [nn.Conv2d(256, 256, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
		# model3 += [nn.ReLU(True), ]
		# model3 += [norm_layer(256), ]
	
		# model4 = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
		# model4 += [nn.ReLU(True),]
		# model4 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
		# model4 += [nn.ReLU(True),]
		# model4 += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
		# model4 += [nn.ReLU(True),]
		# model4 += [norm_layer(256),]

		# model5=[nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),]
		# model5+=[nn.ReLU(True),]
		# model5+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),]
		# model5+=[nn.ReLU(True),]
		# model5+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),]
		# model5+=[nn.ReLU(True),]

		model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
		model1+=[nn.ReLU(True),]
		model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
		model1+=[nn.ReLU(True),]
		model1+=[norm_layer(64),]

		model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
		model2+=[nn.ReLU(True),]
		model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
		model2+=[nn.ReLU(True),]
		model2+=[norm_layer(128),]

		model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
		model3+=[nn.ReLU(True),]
		model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
		model3+=[nn.ReLU(True),]
		model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
		model3+=[nn.ReLU(True),]
		model3+=[norm_layer(256),]

		model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
		model4+=[nn.ReLU(True),]
		model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
		model4+=[nn.ReLU(True),]
		model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
		model4+=[nn.ReLU(True),]
		model4+=[norm_layer(512),]

		model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
		model5+=[nn.ReLU(True),]
		model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
		model5+=[nn.ReLU(True),]
		model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
		model5+=[nn.ReLU(True),]
		model5+=[norm_layer(512),]

		model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
		model6+=[nn.ReLU(True),]
		model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
		model6+=[nn.ReLU(True),]
		model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
		model6+=[nn.ReLU(True),]
		model6+=[norm_layer(512),]

		model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
		model7+=[nn.ReLU(True),]
		model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
		model7+=[nn.ReLU(True),]
		model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
		model7+=[nn.ReLU(True),]
		model7+=[norm_layer(512),]

		model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
		model8+=[nn.ReLU(True),]
		model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
		model8+=[nn.ReLU(True),]
		model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
		model8+=[nn.ReLU(True),]


		model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

		self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)

		self.model1 = nn.Sequential(*model1)
		self.model2 = nn.Sequential(*model2)
		self.model3 = nn.Sequential(*model3)
		self.model4 = nn.Sequential(*model4)
		self.model5 = nn.Sequential(*model5)
		self.model6 = nn.Sequential(*model6)
		self.model7 = nn.Sequential(*model7)
		self.model8 = nn.Sequential(*model8)

		self.softmax = nn.Softmax(dim=1)
		self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')

		self.normalize_l = lambda x : (x - 50)/100
		self.normalize_ab = lambda x : x/110

		self.unnormalize_l = lambda x : (x*100) + 50
		self.unnormalize_ab = lambda x : x*110

		self.norm = lambda x : (x - torch.min(x))/(torch.max(x) - torch.min(x))

	def forward(self, X):
		conv1 = self.model1(self.normalize_l(X))
		conv2 = self.model2(conv1)
		conv3 = self.model3(conv2)
		conv4 = self.model4(conv3)
		conv5 = self.model5(conv4)
		conv6 = self.model6(conv5)
		conv7 = self.model7(conv6)
		logit = self.model8(conv7)
		out = self.softmax(logit)
		return logit, self.upsample(self.unnormalize_ab(out))

	def training_step(self, batch, batch_idx):
		X, y = batch
		target_z, _ = self.sub_model_eccv.get_class_forward(X)
		pred_z, pred_y = self.forward(X)

		loss = self.mcent_loss(pred_z, target_z)
		loss2 = F.mse_loss(pred_y, y)
		self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
		return self.loss_weights[0]*loss + self.loss_weights[1]*loss2

	def validation_step(self, batch, batch_idx):
		X, y = batch
		target_z, _ = self.sub_model_eccv.get_class_forward(X)
		pred_z, pred_y = self.forward(X)

		loss = self.mcent_loss(pred_z, target_z)
		loss2 = F.mse_loss(pred_y, y)
		self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
		return self.loss_weights[0]*loss + self.loss_weights[1]*loss2

	def test_step(self, batch, batch_idx):
		X, y = batch
		target_z, _ = self.sub_model_eccv.get_class_forward(X)
		pred_z = self.forward(X)
		loss = self.loss3_user(pred_z, target_z)
		self.log('test_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
		return loss

	def annealed_mean(self, Z):
		batch_size, H, W = Z.shape[0], Z.shape[2], Z.shape[3]
		T = .38
		a = torch.exp(torch.log(Z)/T).view(batch_size, (H*W), -1)
		b = torch.sum(torch.exp(torch.log(Z.view(batch_size, (H*W), -1)/T)), dim = 2)
		Y = torch.mean(a, dim = 2)/b
		Y = Y.reshape(batch_size, H, W)
		Y = Y[:,None,:,:]
		return Y

	# multinomial cross entropy loss
	def mcent_loss(self, pred_z, target_z):
		batch_size, H, W = pred_z.shape[0], pred_z.shape[2], pred_z.shape[3]
		argmx = torch.argmax(target_z, dim = 1).flatten()
		w = torch.zeros(size=argmx.size()).flatten()
		w = self.prior_probs[argmx[:]]
		w = w.reshape(batch_size, -1)
		sigma = torch.sum(target_z.view(batch_size, (H*W), -1) * torch.log((pred_z + torch.finfo(pred_z.data.dtype).eps).view(batch_size, (H*W), -1)), dim=2)
		loss = -torch.sum(w * sigma, dim = 1)
		return torch.mean(loss)

	def configure_optimizers(self):
		# 0.001, 0.01, 0.1, 1, and 10.
		optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=opt.betas, weight_decay=1e-3) #1e-5
		# T_max should be number of cycles to vary the learning rate, i set to 3 (12,000 steps if batch size is 25)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=1e-7,
															   T_max=self.T_max)  # TODO comment out if you don't want to mess with
		return [optimizer], [scheduler]

	def predict_step(self, batch: int, batch_idx: int, dataloader_idx: int = None):
		return self(batch)

	def print_photos(self, epoch):
		device = torch.device('cuda:0')
		for i in range(4):
			name = f'./test_images_for_training/test{i}.JPEG'
			im_test = util.load_img(name)
			X_test, _ = util.preprocess_img(im_test)
			X_test = X_test[None,:,:,:]
			X_test = X_test.to(device)
			_, y_pred = self.forward(X_test)
			im_test = util.postprocess_tens(X_test,  y_pred)
			util.save_im(f'./image_log/test_{i}_{epoch}.JPEG', im_test)

		for i in range(3):
			name = f'./test_images_for_training/train{i}.JPEG'
			im_test = util.load_img(name)
			X_test, _ = util.preprocess_img(im_test)
			X_test = X_test[None,:,:,:]
			X_test = X_test.to(device)
			_, y_pred = self.forward(X_test)
			im_test = util.postprocess_tens(X_test,  y_pred)
			util.save_im(f'./image_log/train_{i}_{epoch}.JPEG', im_test)
		
		im_big = util.load_img('./test_images_for_training/big_image.jpg')
		X_org, X_big = util.preprocess_img_org(im_big)
		X_big = X_big.to(device)
		X_big = X_big
		_, y_pred = self.forward(X_big)
		im_big = util.postprocess_tens(X_big,  y_pred)
		util.save_im(f'./image_log/big_{epoch}.JPEG', im_big)

	def on_epoch_end(self):
		global_step = self.global_step
		# epoch = self.current_epoch
		self.print_photos(epoch)
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
def run_test():
	checkpoint =  torch.load('./logs/default/version_127/checkpoints/last.ckpt')

def run_trainer():
	early_stop_call_back = EarlyStopping(
		monitor='val_loss_epoch',
		min_delta=0.00,
		check_finite=True,
		patience=7,
		verbose=True,
		check_on_train_epoch_end=True,
		mode='min'
	)
	'''log learning rate'''
	lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

	checkpoint_callback = ModelCheckpoint(
		save_last=True,
		verbose=True,
		every_n_val_epochs=1
	)
	model2 = model_eccv16.eccv16(pretrained=True)
	# model3 = model_siggraph.siggraph17(pretrained=True)
	l = 'L2' #'RarityWeighted'
	max_epochs = 100
	batch_size = 64
	num_workers = 6
	T_max = np.floor(100000/batch_size)*max_epochs
	model = Colorization_model_Reduced(sub_model = (model2, True), loss_weights = (3/4, 1/4), loss=l, num_workers = num_workers, batch_size=batch_size, T_max=T_max)  # TODO set loss as RarityWeighted or L2, default: L2
	logger = loggers.TensorBoardLogger(save_dir='logs/')
	if torch.cuda.is_available():
		print("using GPU")
		num_gpus = torch.cuda.device_count()
	else:
		print("using CPU")
		num_gpus=0
	trainer = Trainer(
					  max_epochs=100,
					  gpus=num_gpus,
					  logger=logger,  # use default tensorboard
					  log_every_n_steps=20,  # log every update step for debugging
					  limit_train_batches=1.0,
					  limit_val_batches=1.0,
					  check_val_every_n_epoch=1,
					  callbacks=[early_stop_call_back, lr_callback, checkpoint_callback])
	trainer.fit(model)



if __name__ == '__main__':
	# start trainer
	run_trainer()
