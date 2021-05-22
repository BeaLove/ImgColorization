from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

list = dir(models)

print(list)

shuffle = models.shufflenet_v2_x0_5(pretrained=True, progress=True)

print(shuffle)