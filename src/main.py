# import libraries
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from multiprocessing import Process

import data_loader as dl

import torch.nn as nn



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
opt = parser.parse_args()




class Colorization_model(pl.LightningModule):

	def __init__(self, norm_layer=nn.BatchNorm2d):
		super(Colorization_model, self).__init__()
		self.data_loaders = dl.return_loaders()

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


	def forward(self, x):
		#forward
		return 0

	def training_step(self, batch, batch_idx):
		pass
	
#     def validation_step(self,batch,batch_idx):
		


#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2),weight_decay=1e-5)

	# @pl.data_loader
	def train_dataloader(self):
		return self.data_loaders['train']


	# @pl.data_loader
	def test_dataloader(self):
		return self.data_loaders['test']

	def configure_optimizers(self):
		# Dummy optimizer for testing
		return torch.optim.Adam(self.parameters(), lr=1e-5)


def run_trainer():
	model = Colorization_model()
	trainer = Trainer(max_epochs=1)
	trainer.fit(model)

if __name__ == '__main__':
	# p1 = Process(target=run_trainer)                    # start trainer
	run_trainer()

	#p1.start()
	#p2 = Process(target=run_tensorboard(new_run=True))  # start tensorboard
	#p2.start()
	#p1.join()
	#p2.join()