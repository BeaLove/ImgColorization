import torch
from pathlib import Path
from functools import partial


from PIL import Image
import numpy as np
import sklearn.neighbors as knn
from skimage import io
from torchvision.transforms import Normalize
#import misc.npy_loader.loader as npy

'''import loss to debug'''

#POINTS_IN_HULL = npy.load('authors_pts_in_hull')
bins_centers = np.load('../npy/bins.npy')
'''don't reinvent the wheel!'''
class Dataset(torch.utils.data.Dataset):
	def __init__(self, dataset, soft_encoding = True):
		self.dataset = dataset
		self.kernels = bins_centers
		self.L_normalize = 100
		self.kernel_normalize = np.max(np.abs(self.kernels))
		self.num_bins = len(self.kernels)
		self.neighborhood = knn.NearestNeighbors(n_neighbors=5).fit(self.kernels/self.kernel_normalize)
		self.soft_encoding = soft_encoding


	def __getitem__(self, i):

		#with sklearn
		im = io.imread(str(self.dataset[i]))
		X, Y = im[np.newaxis, :, :, 0], im[:, :, 1:]
		X = torch.tensor(X, dtype=torch.float32)/self.L_normalize

		if self.soft_encoding:
			Y = torch.tensor(self.softEncoding(Y, sigma=5))
		else:
			Y = torch.tensor(np.moveaxis(Y, 2, 0))/self.kernel_normalize #switch to channel first format
		return X, Y
		
	def __len__(self):
		return len(self.dataset)

	def softEncoding(self, pixels, sigma=5):
		'''args: image a,b channels H*W
			Gets 5 nearest neighbors of the quantized bins in output space (313)
			weighted by Gaussian kernel, sigma=5, See Colorful Image Colorization Zhang, Isola, Efros
			returns: soft-encoded target matrix H*W*313'''
		w = pixels.shape[0]
		h = pixels.shape[1] #height of an image is
		dist, indices = self.neighborhood.kneighbors(pixels.reshape(h*w, 2)/self.kernel_normalize)
		weights = np.exp(-(dist ** 2) / 2 * sigma ** 2)
		weights = weights / np.sum(weights, axis=1, keepdims=True)
		'''check weights sum to 1'''
		#sum_ = np.sum(weights, axis=1, keepdims=True)
		target_vector = np.zeros((h*w, self.num_bins))
		for i in range(len(weights)):
			target_vector[i, indices[i]] = weights[i]
		#test_sum = np.sum(target_vector, axis=1)
		target_vector = target_vector.reshape(self.num_bins, h, w)
		#test_sum2 = np.sum(target_vector, axis=2)
		return target_vector

def prepare(set_spec, params):
	''' params = (batch_size, num_workers, shuffle) '''
	X = list(set_spec.glob('**/*.TIF'))
	batch_size, num_workers, shuffle, soft_encoding = params
	dataset = Dataset(X, soft_encoding)

	train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
											   shuffle = shuffle, drop_last=False)
	
	return train_loader

def return_loaders(batch_size = 64, num_workers = 2, shuffle = True, soft_encoding=True):
	paths = ['../dataset/test_tif', '../dataset/train_tif', '../dataset/val_tif']
	#paths = ['../dataset/test_tif', '../dataset/train_tif', '../dataset/val_tif']
	paths = map(Path, paths)

	dataset = {}
	prepare_partial = partial(prepare, params = (batch_size, num_workers, shuffle, soft_encoding))
	dataset['test'], dataset['train'], dataset['validation'] = map(prepare_partial, paths)
	
	return dataset




