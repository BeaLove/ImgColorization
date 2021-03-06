import os
from skimage.color.colorconv import lab2rgb
import torch
import numpy as np
import util
from skimage import io, color
from main import Colorization_model
from main_mini import Colorization_model_Reduced
from pytorch_lightning import LightningModule

def _decode_mode(data):
	'''in: data: torch tensor of ab_channels
		operation: selects the mode pixel value from color bins
		otu'''
	import warnings
	warnings.warn("Build not passing")
	data = data[0, :, :, :]
	(Q, H, W) = data.shape
	y_idx = torch.argmax(data, dim=0, keepdim=False)
	y = torch.index_select(torch.tensor(util.BIN_CENTERS), 0, y_idx.reshape(-1)).reshape(2, H,W,)
	a = y[:,:,0]
	b = y[:,:,1]
	return y.cpu().detach().numpy()

def _decode_mean(data):
	data = data[0, :, :, :]
	(Q, H, W) = data.shape
	data = data.cpu().detach().numpy()
	out = np.zeros((2, H, W))
	for x in range(H):
		for y in range(W):
			q = data[:,x,y]
			top_5 = np.argsort(q)[:5]
			colors = util.BIN_CENTERS[top_5]
			mean = np.mean(colors, axis=0)
			out[:,x,y] = mean
	#y_idx = np.nonzero(data)[2]
	return out

def _decode_annealing(data):
	'''annealed mean color decoding
		in: data tensor (268, 64, 64)
		out: annealed mean color prediction
		'''
	T = .38
	data = data[0,:,:,:]
	(Q,H,W) = data.shape
	data = data.cpu().detach().numpy()
	out = np.zeros((2,H,W))
	for x in range(H):
		for y in range(W):
			q = data[:,x,y]
			sum = np.sum(np.exp(q/T))
			annealed= np.exp(q/T)/sum
			top_5 = np.argsort(annealed)[:5]
			colors = util.BIN_CENTERS[top_5]
			mean = np.mean(colors, axis=0)
			out[:,x,y] = mean
	#y_idx = np.nonzero(annealed)[2]

	return out


def decode_targets(data, algorithm = 'annealing'):
	""" 
		Takes Z and returns point estimate Y
		algorithm in {'annealing', 'mode', 'mean'}
	"""

	if algorithm == 'annealing':
		return _decode_annealing(data)
	elif algorithm == 'mode':
		return _decode_mode(data)
	elif algorithm == 'mean':
		return _decode_mean(data)
	else:
		raise ValueError(f'algorithm = {algorithm} not a valid argument')

def load_and_decode(img_path, last_checkpoint_path, model_name = None, resize=False, algorithm= 'annealing'):
	'''in: img_path string, model: colorizer model name,
		args: 'annealing', 'mean', or 'mode', type of decoding
		loads a color image, splits channels and colorizes the bw channel from custom loss function model prediction
		out: colorized img'''
	img = util.load_image_raw(img_path, resize=False)

	'''optional using Lightning'''
	model = Colorization_model_Reduced()
	pretrained_model = model.load_from_checkpoint(last_checkpoint_path)
	pretrained_model.eval()
	#TODO is this the correct way to load a trained model?
	L, a, b = util.split_channels(img)
	L = torch.tensor(L/100, dtype=torch.float32)

	prediction = pretrained_model(L)

	ab_channels = decode_targets(prediction, algorithm=algorithm)

	im_stitched = util.stich_image(L[0,:,:,:]*100, ab_channels)
	im_decoded = np.round(util.data2rgb(im_stitched)*255)
	io.imshow(im_decoded)
	io.show()
	os.makedirs('outputs', exist_ok=True)
	save_path = path.split(sep = '/')[-1]
	io.imsave('outputs/'+algorithm+save_path, im_decoded)



if __name__ == '__main__':
	path = 'input_img/test_color_img.JPEG'
	#path = 'input_img/test_tif_0.TIF'
	chkpt_path = 'version_93/epoch=8-step=7037.ckpt'
	model = 'trained_models/ColorizationModelOverfitTest.pth'
	load_and_decode(path, chkpt_path, model_name=model, algorithm='annealing', resize=False)
	X, Y, im = util.load_image_softencoded(path)

	bc = torch.tensor(util.BIN_CENTERS)
	'''compare values'''
	ab_channels = decode_targets(Y, algorithm = 'annealing')
	L = X.detach().cpu().numpy().reshape(64, 64)
	im_stitched = util.stich_image(L, ab_channels)
	im_decoded_annealed = util.data2rgb(im_stitched)

	ab_channels = decode_targets(Y, algorithm = 'mode')
	L = X.detach().cpu().numpy().reshape(64, 64)
	# im_raw = util.load_image_raw(path, resize = True)
	im_stitched = util.stich_image(L, ab_channels)
	im_decoded_mode = util.data2rgb(im_stitched)

	ab_channels = decode_targets(Y, algorithm = 'mean')
	L = X.detach().cpu().numpy().reshape(64, 64)
	# im_raw = util.load_image_raw(path, resize = True)
	im_stitched = util.stich_image(L, ab_channels)
	im_decoded_mean = util.data2rgb(im_stitched)
	diff_mode_mean = im_decoded_mode - im_decoded_mean
	diff_mean_annealed = im_decoded_mean - im_decoded_annealed

	io.imshow(im_decoded_annealed)
	os.makedirs('outputs', exist_ok=True)
	io.imsave('outputs/decoded_annealed_test.jpg', im_decoded_annealed)
	io.show()
	# im_decoded = util.data2lab(im_decoded)