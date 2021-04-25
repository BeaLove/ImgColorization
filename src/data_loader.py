import torch
from pathlib import Path
from functools import partial
from PIL import Image
import numpy as np

class Dataset(torch.utils.data.Dataset):
	def __init__(self, dataset):
		self.dataset = dataset

	def __getitem__(self, i):
		im = Image.open(self.dataset[i])
		im = np.array(im, dtype = np.float32)
		X, y = im[:,:,0], im[:,:,1:]
		X, y = X/255, y/255 # Should we normalize it like this here?
		return X, y
		
	def __len__(self):
		return len(self.dataset)

def prepare(set_spec, params):
	''' params = (batch_size, num_workers, shuffle) '''
	X = list(set_spec.glob('**/*.TIF'))

	dataset = Dataset(X)

	batch_size, num_workers, shuffle = params
	train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
	
	return train_loader

def return_loaders(batch_size = 25, num_workers = 0, shuffle = True):
	paths = ['../dataset/test_tif', '../dataset/train_tif', '../dataset/val_tif']
	paths = map(Path, paths)

	dataset = {}
	prepare_partial = partial(prepare, params = (batch_size, num_workers, shuffle))
	dataset['test'], dataset['train'], dataset['validation'] = map(prepare_partial, paths)
	
	return dataset
