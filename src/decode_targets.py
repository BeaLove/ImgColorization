from skimage.color.colorconv import lab2rgb
import torch
import numpy as np
import util
from skimage import io, color

def _decode_mode(data):
	import warnings
	warnings.warn("Build not passing")

	(H, W, Q) = data.shape
	y = np.zeros(shape=(H, W, 2))

	for h in range(H):
		for w in range(W):
			value, index = torch.mode(data[h, w, :])
			y[h, w] = util.BIN_CENTERS[index.item()]
	a = y[:,:,0]
	b = y[:,:,1]
	return (a, b)

def _decode_mean(data):
	raise ValueError('Not implemented')
	return -1

def _decode_annealing(data):
	T = .38
	raise ValueError('Not implemented')
	return -1

def decode_targets(data, args = 'annealing'):
	""" 
		Build: Not passing
		args in {'annealing', 'mode', 'mean'}
	"""

	if args == 'annealing':
		return _decode_annealing(data)
	elif args == 'mode':
		return _decode_mode(data)
	elif args == 'mean':
		return _decode_mean(data)
	else:
		raise ValueError(f'args = {args} not a valid argument')

if __name__ == '__main__':
	path = 'test_color_image.jpg'
	X, Y, im = util.load_image_softencoded(path)

	bc = util.BIN_CENTERS

	a, b = decode_targets(Y, args = 'mode')
	L = X.detach().cpu().numpy().reshape(64, 64)

	im_raw = util.load_image_raw(path, resize = True)
	im_decoded = util.stich_image(L, a, b)
	# im_decoded = util.data2lab(im_decoded)