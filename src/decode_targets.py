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
	y_idx = torch.argmax(data, dim=2, keepdim=False)
	y = torch.index_select(torch.tensor(util.BIN_CENTERS), 0, y_idx.reshape(-1)).reshape(H,W,2)
	a = y[:,:,0]
	b = y[:,:,1]
	return y.cpu().detach().numpy()

def _decode_mean(data):
	data = data[0, :, :, :]
	(Q, H, W) = data.shape
	data = data.cpu().detach().numpy()
	y_idx = np.nonzero(data)[2]
	q_dim_shape = int(y_idx.size/(64*64))
	y = util.BIN_CENTERS[y_idx]
	y = y.reshape(H,W,q_dim_shape,2)
	mean = np.mean(y,axis=2)
	y = y_idx.reshape(H, W, q_dim_shape)
	return mean

def _decode_annealing(data):
	'''annealed mean color decoding
		in: data tensor (64,64,441)
		'''
	T = .38
	data = data[0,:,:,:]
	(Q, H, W) = data.shape
	data = data.cpu().detach().numpy()
	annealed = np.zeros((H, W, Q))
	for x in range(H):
		for y in range(W):
			q = data[x,y]
			sum = np.sum(np.exp(np.log(q/T)))
			annealed[:,x, y] = np.where(q > 0, np.exp(np.log(q/T))/sum, 0)
	y_idx = np.nonzero(annealed)[2]
	q_dim_shape = int(y_idx.size / (64 * 64))
	y = util.BIN_CENTERS[y_idx]
	y = y.reshape(q_dim_shape, H, W, 2)
	annealed_mean = np.mean(y, axis=0)
	return annealed_mean


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
	'''in: img_path string, model: colorizer model name, args: 'annealing', 'mean', or 'mode', type of decoding
		loads img, splits channels and colorizes the bw
		out: colorized img'''
	#img = util.load_image_raw(img_path, resize=False)
	img = io.imread(img_path)
	'''optional using Lightning'''
	model = Colorization_model_Reduced()
	pretrained_model = model.load_from_checkpoint(last_checkpoint_path)
	pretrained_model.eval()
	#TODO is this the correct way to load a trained model?
	L, a, b = util.split_channels(img)
	L = torch.tensor(L/100, dtype=torch.float32)
	#Y = colorizer.predict(L)
	prediction = pretrained_model(L)
	ab_channels = decode_targets(prediction, algorithm=algorithm)
	im_stitched = util.stich_image(L[0,:,:,:], ab_channels)
	im_decoded = util.data2rgb(im_stitched)
	os.makedirs('outputs', exist_ok=True)
	io.imsave(os.path.join('outputs', img_path), im_decoded)



if __name__ == '__main__':
	path = 'input_img/test_color_image.jpg'
	path = 'input_img/test_tif_0.TIF'
	chkpt_path = 'version_93/epoch=8-step=7037.ckpt'
	model = 'trained_models/ColorizationModelOverfitTest.pth'
	load_and_decode(path, chkpt_path, model_name=model, resize=False)
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