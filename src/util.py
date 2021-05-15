from PIL import Image
import numpy as np
from skimage import color, io
import torch
import sklearn.neighbors as knn
import torch.nn.functional as F
from IPython import embed
import cv2

POINTS_IN_HULL = np.load('authors_pts_in_hull.npy')
BIN_CENTERS = np.load('bin_centers.npy')

def load_image(path, shape = (64, 64), resize = False):
	""" loads an image given a path 
			resize to shape
			in: path as a string, resize = True if you want to resize the image to shape = (height, width)
			out: (img, img_raw)
	"""
	im_raw = io.imread(path)
	
	if resize:
		im_raw = resize_img(im_raw, shape = shape)
	
	im_lab = color.rgb2lab(im_raw)/255
	return im_lab, im_raw

def stich_image(L, a, b):
	assert L.shape == a.shape == b.shape # sanity check
	out = np.ones(shape = (L.shape[0], L.shape[1], 3))
	out[:,:,0], out[:,:,1], out[:,:,2] = L, a, b # This is a bit stupid but np.array gave me (3, 64, 64) and that is not compatible with skimage
	# out = np.array(list(zip(L, a, b)))
	return out

def load_image_raw(path, resize = False):
	""" loads the raw image from path as lab """
	if resize:
		return color.rgb2lab(resize_img(io.imread(path)))
	else:
		return color.rgb2lab(io.imread(path))

def data2lab(im):
	""" reverts our encoded values back to proper lab as defined by skimage """
	raise Exception("Not implemented")
	return color.lab2rgb(im)

def resize_img(im, shape = (64, 64)):
	""" This resizes the image given our datastructure """
	im_resized = cv2.resize(im, dsize=(shape[0], shape[1]), interpolation=cv2.INTER_LINEAR)
	return im_resized

def load_image_softencoded(path, resize = True):
	""" This will load one image exactly as the current dataloader given a path to a jpg. """
	im, _ = load_image(path, resize = resize)

	X, Y = im[np.newaxis, :, :, 0], im[:, :, 1:]
	X = torch.tensor(X, dtype=torch.float32)
	Y = torch.tensor(_softEncoding(Y, sigma=5))
	return X, Y, im

def _softEncoding(pixels, sigma=5):
	'''args: image a,b channels H*W
		Gets 5 nearest neighbors of the quantized bins in output space (313)
		weighted by Gaussian kernel, sigma=5, See Colorful Image Colorization Zhang, Isola, Efros
		returns: soft-encoded target matrix H*W*313'''
	num_bins = len(BIN_CENTERS)
	kernel_normalize = np.max(np.abs(BIN_CENTERS))
	w = pixels.shape[0]
	h = pixels.shape[1] #height of an image is
	ngh = knn.NearestNeighbors(n_neighbors=5).fit(BIN_CENTERS/kernel_normalize)
	dist, indices = ngh.kneighbors(pixels.reshape(h*w, 2)/kernel_normalize)
	weights = np.exp(-(dist ** 2) / 2 * sigma ** 2)
	weights = weights / np.sum(weights, axis=1, keepdims=True)
	'''check weights sum to 1'''
	sum_ = np.sum(weights, axis=1, keepdims=True)
	target_vector = np.zeros((h*w, num_bins))
	for i in range(len(weights)):
		target_vector[i, indices[i]] = weights[i]
	test_sum = np.sum(target_vector, axis=1)
	target_vector = target_vector.reshape(w, h,num_bins)
	test_sum2 = np.sum(target_vector, axis=2)
	return target_vector