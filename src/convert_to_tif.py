from pathlib import Path
from PIL import Image, ImageOps, ImageCms
from tqdm import tqdm
import os


""" 
Extract the dataset (folders test, train, val) and place them into folder dataset.
Then run this script to convert the .JPEG images to .TIF images.

Call this function in a terminal rooted in src/ folder otherwise no cigar.
Since the paths assumes your terminals pwd is at */IMGCOLORIZATION/src

https://stackoverflow.com/questions/52767317/how-to-convert-rgb-image-pixels-to-lab/53353542#53353542
"""


def convert_to_tif():
	os.makedirs('../dataset/train_tif/', exist_ok = True)
	os.makedirs('../dataset/val_tif/', exist_ok = True)
	os.makedirs('../dataset/test_tif/', exist_ok = True)
	paths = ['../dataset/test/', '../dataset/val/', '../dataset/train/']
	
	for path in paths:
		path = Path(path)
		pic_list = list(path.glob('**/*.JPEG'))

		srgb_p = ImageCms.createProfile("sRGB")
		lab_p  = ImageCms.createProfile("LAB")
		rgb_to_lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")

		for pic in tqdm(pic_list):
			with Image.open(pic).convert('RGB') as im:
				tif_img = ImageCms.applyTransform(im, rgb_to_lab)
				word = str(pic).split('/')[2] if os.name != 'nt' else str(pic).split('\\\\')[2]
				save_path = str(pic).replace(word+'/' , word + '_tif/')
				save_path = save_path.replace('.JPEG', '.TIF')
				save_path_dir = ''.join([w+'/' for w in save_path.split('/')[:-1]])
				os.makedirs(save_path_dir, exist_ok = True)
				tif_img.save(save_path, 'TIFF')

if __name__ == '__main__':
	convert_to_tif()