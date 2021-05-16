# import libraries
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from multiprocessing import Process
from loss import RarityWeightedLoss, PRIOR_PROBS

import data_loader as dl

import torch.nn as nn



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
opt = parser.parse_args()

"""
 Code found from: https://github.com/richzhang/colorization
"""


class Colorization_model(pl.LightningModule):
	def __init__(self, norm_layer=nn.BatchNorm2d, lamda=0.5):
		super(Colorization_model, self).__init__()
		self.data_loaders = dl.return_loaders()
		self.loss_criterion = RarityWeightedLoss(PRIOR_PROBS, lamda= lamda, num_bins=313)

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

		self.model1 = nn.Sequential(*model1)
		self.model2 = nn.Sequential(*model2)
		self.model3 = nn.Sequential(*model3)
		self.model4 = nn.Sequential(*model4)
		self.model5 = nn.Sequential(*model5)
		self.model6 = nn.Sequential(*model6)
		self.model7 = nn.Sequential(*model7)
		self.model8 = nn.Sequential(*model8)

		self.softmax = nn.Softmax(dim=1)
		self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
		self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')


	def forward(self, X):
		conv1_2 = self.model1(X)
		conv2_2 = self.model2(conv1_2)
		conv3_3 = self.model3(conv2_2)
		conv4_3 = self.model4(conv3_3)
		conv5_3 = self.model5(conv4_3)
		conv6_3 = self.model6(conv5_3)
		conv7_3 = self.model7(conv6_3)
		conv8_3 = self.model8(conv7_3)
		out_reg = self.model_out(self.softmax(conv8_3))
		return out_reg

	def training_step(self, batch, batch_idx):
		X, y = batch
		output = self.forward(X)
		loss = self.loss_criterion(output, y)
		self.log('train_loss', loss)
		return loss

	def validation_step(self,batch,batch_idx):
		X, y = batch
		output = self.forward(X)
		loss = self.loss_criterion(output, y)
		self.log('val_loss', loss)
		return loss
		


#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2),weight_decay=1e-5)

	# @pl.data_loader
	def train_dataloader(self):
		return self.data_loaders['train']

	# @pl.data_loader
	def test_dataloader(self):
		return self.data_loaders['test']
	
	def val_dataloader(self):
		return self.data_loaders['validation']

	def configure_optimizers(self):
		# Dummy optimizer for testing
		return torch.optim.Adam(self.parameters(), lr=1e-5)


def run_trainer():
	early_stop_call_back = EarlyStopping(
		monitor='val_loss',
		min_delta=0.00,
		patience=5,
		verbose=False,
		mode='max'
	)
	model = Colorization_model(lamda=0.5)
	trainer = Trainer(max_epochs=1,
					  limit_train_batches=0.05,
					  limit_val_batches=1.0,
					  limit_test_batches=1.0)
	trainer.fit(model)
	os.makedirs('trained_models', exist_ok=True)
	name = 'ColorizationModelOverfitTest.pth'
	torch.save(model.state_dict(), os.path.join('trained_models', name))

if __name__ == '__main__':
	# p1 = Process(target=run_trainer)                    # start trainer
	run_trainer()

	#p1.start()
	#p2 = Process(target=run_tensorboard(new_run=True))  # start tensorboard
	#p2.start()
	#p1.join()
	#p2.join()