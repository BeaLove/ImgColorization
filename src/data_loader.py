import torch
from pathlib import Path
from functools import partial
from PIL import Image
import numpy as np
import sklearn.neighbors as knn

POINTS_IN_HULL = np.load('pts_in_hull (1).npy')
'''don't reinvent the wheel!'''
class Dataset(torch.utils.data.Dataset):
	def __init__(self, dataset):
		self.dataset = dataset
		self.kernels = POINTS_IN_HULL
		self.neighborhood = knn.NearestNeighbors(n_neighbors=5).fit(self.kernels/110)


	def __getitem__(self, i):
		im = Image.open(self.dataset[i])
		im = np.array(im, dtype = np.float32)
		X, Y = im[:,:,0], im[:,:,1:]
		'''zip Y together as tuple (a,b) values'''
		'''histogram of (a,b) values'''
		'''5-nearest neighbors encoding 
		with gaussian kernel sigma=5 (scipy?)'''
		X, Y = X, Y
		Y = self.softEncoding(Y, sigma=5)
		return X, Y
		
	def __len__(self):
		return len(self.dataset)

	def softEncoding(self, pixels, sigma=5):
		dist, indices = knn.kneighbors(pixels.reshape(pixels.shape[0] * pixels.shape[1], 2))
		weights = np.exp(-(dist ** 2) / 2 * sigma ** 2)
		weights = weights / np.sum(weights, axis=1, keepdims=True)
		'''check weights sum to 1'''
		#sum_ = np.sum(weights, axis=1, keepdims=True)
		target_vector = np.zeros((100, 313))
		for i in range(len(weights)):
			target_vector[i, indices[i]] = weights[i]
		#test_sum = np.sum(target_vector, axis=1)
		target_vector = target_vector.reshape(10, 10, 313)
		#test_sum2 = np.sum(target_vector, axis=2)
		return target_vector

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
