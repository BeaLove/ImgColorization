import torch
from pathlib import Path
from functools import partial


import numpy as np
import sklearn.neighbors as knn
from skimage import io, color


'''import loss to debug'''
from loss import RarityWeightedLoss, PRIOR_PROBS

POINTS_IN_HULL = np.load('authors_pts_in_hull.npy')
bins_centers = np.load('bin_centers.npy')
'''don't reinvent the wheel!'''
class Dataset(torch.utils.data.Dataset):
	def __init__(self, dataset, soft_encoding = True):
		self.dataset = dataset
		self.kernels = bins_centers
		self.kernel_normalize = np.max(np.abs(self.kernels))
		self.num_bins = len(self.kernels)
		self.neighborhood = knn.NearestNeighbors(n_neighbors=5).fit(self.kernels/self.kernel_normalize)
		self.soft_encoding = soft_encoding


	def __getitem__(self, i):
		#im = Image.open(self.dataset[i])
		#with sklearn

		im = io.imread(str(self.dataset[i]))
		#im.show()
		#im = torchvision.transforms.ToTensor()(im)
		#im = torch.tensor(im, dtype=)
		#im = np.array(im, dtype = np.float64)
		#with skimage
		X, Y = im[np.newaxis, :, :, 0], im[:, :, 1:]
		#X, Y = im[np.newaxis, :,:, 0]/255*100, im[:,:,1:]-127
		X = torch.tensor(X, dtype=torch.float32)
		'''plt.imshow(X)
		plt.show()
		plt.imshow(Y[1,:,:])
		plt.show()'''
		#transforms.ToPILImage()(X).show()
		#transforms.ToPILImage()(Y[1,:,:]).show()
		'''return X, Y as tensors'''
		#X, Y = X, Y
		if self.soft_encoding:
			Y = torch.tensor(self.softEncoding(Y, sigma=5))
		else:
			Y = torch.tensor(Y)
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
		sum_ = np.sum(weights, axis=1, keepdims=True)
		target_vector = np.zeros((h*w, self.num_bins))
		for i in range(len(weights)):
			target_vector[i, indices[i]] = weights[i]
		test_sum = np.sum(target_vector, axis=1)
		target_vector = target_vector.reshape(w, h, self.num_bins)
		test_sum2 = np.sum(target_vector, axis=2)
		return target_vector

def prepare(set_spec, params):
	''' params = (batch_size, num_workers, shuffle) '''
	X = list(set_spec.glob('**/*.TIF'))

	dataset = Dataset(X)
	'''test code for soft encoding'''
	''''test code for loss comment out before training'''
	'''loss_crit = RarityWeightedLoss(PRIOR_PROBS, lamda = 0.5, num_bins=dataset.num_bins)
	sample, target = dataset.__getitem__(0)
	sample2, target2 = dataset.__getitem__(0)'''
	batch_size, num_workers, shuffle = params
	train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
	
	return train_loader

def return_loaders(batch_size = 25, num_workers = 0, shuffle = True):
	#for sklearn:
	paths = ['../dataset/test_tif', '../dataset/train_tif', '../dataset/val_tif']
	#paths = ['../dataset/test_tif', '../dataset/train_tif', '../dataset/val_tif']
	paths = map(Path, paths)

	dataset = {}
	prepare_partial = partial(prepare, params = (batch_size, num_workers, shuffle))
	dataset['test'], dataset['train'], dataset['validation'] = map(prepare_partial, paths)
	
	return dataset

'''test code for soft encoding'''
dataset = return_loaders()



