from pathlib import Path
from PIL import Image, ImageOps, ImageCms
from tqdm import tqdm
import os
from skimage import io, color


""" 
Extract the dataset (folders test, train, val) and place them into folder dataset.
Then run this script to convert the .JPEG images to .TIF images.

https://stackoverflow.com/questions/52767317/how-to-convert-rgb-image-pixels-to-lab/53353542#53353542
"""


def convert_to_tif():
	root = os.path.dirname(os.path.abspath(__file__)) + '/'
	os.makedirs(root + '../dataset/train_tif/', exist_ok = True)
	os.makedirs(root + '../dataset/val_tif/', exist_ok = True)
	os.makedirs(root +'../dataset/test_tif/', exist_ok = True)
	paths = [root + '../dataset/test/', root + '../dataset/val/', root + '../dataset/train/']
	skip_count = 0
	for path in paths:
		pic_list = []
		for dir, subdir, files in os.walk(path, topdown=False, followlinks=False):
			for f in files:
				pic_list.append(dir + "/" + f)

		for pic in tqdm(pic_list):
			try:
				im = io.imread(pic)
			except ValueError:
				skip_count += 1
				continue
			#with io.imread(pic) as im:
		   #print('hello world')
			try:
				tif_img = color.rgb2lab(im/255)
			except ValueError:
				skip_count += 1
				continue

			pic_str = str(pic)


			word = pic_str.replace(root, '').split('/')[2]
			save_path = pic_str.replace(word, word + '_tif')
			save_path = save_path.replace('.JPEG', '.TIF')
			save_path_dir = ''.join([w + '/' for w in save_path.split('/')[:-1]])
			os.makedirs(save_path_dir, exist_ok=True)
			io.imsave(save_path, tif_img)
	print(f'Skipped: {skip_count}')

if __name__ == '__main__':
	convert_to_tif()