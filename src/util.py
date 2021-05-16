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
	
	im_lab = color.rgb2lab(im_raw/255)
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


""" Don't use the below functions for our datastructure """

def load_img_src(img_path):
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np

def resize_img_src(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def get_result_PSNR(self, result=-1, return_SE_map=False):
	if np.array((result)).flatten()[0] == -1:
		cur_result = self.get_img_forward()
	else:
		cur_result = result.copy()
	SE_map = (1. * self.img_rgb - cur_result)**2
	cur_MSE = np.mean(SE_map)
	cur_PSNR = 20 * np.log10(255. / np.sqrt(cur_MSE))
	if return_SE_map:
		return(cur_PSNR, SE_map)
	else:
		return cur_PSNR

def preprocess_img_src(img_rgb_orig, HW=(256,256), resample=3):
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img_src(img_rgb_orig, HW=HW, resample=resample)
	
	img_lab_orig = color.rgb2lab(img_rgb_orig)
	img_lab_rs = color.rgb2lab(img_rgb_rs)

	img_l_orig = img_lab_orig[:,:,0]
	img_l_rs = img_lab_rs[:,:,0]

	tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
	tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

	return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))