""" to do os independent """

from pathlib import Path
from unicodedata import name
from tqdm import tqdm
from skimage import io, color
from collections import namedtuple


def convert_to_tif():
	root = Path(__file__).parent.absolute()
	tif_test_dir = root / '..' / '..' / 'dataset' / 'test_tif'
	tif_train_dir = root / '..' / '..' / 'dataset' / 'train_tif'
	tif_val_dir = root / '..' / '..' / 'dataset' / 'val_tif'

	Path.mkdir(tif_test_dir, exist_ok=True)
	Path.mkdir(tif_train_dir, exist_ok=True)
	Path.mkdir(tif_val_dir, exist_ok=True)

	source_test = root / '..' / '..' / 'dataset' / 'test'
	source_train = root / '..' / '..' / 'dataset' / 'train'
	source_val = root / '..' / '..' / 'dataset' / 'val'

	source_dirs = {'test': source_test, 'train': source_train, 'val': source_val}

	skip_count = 0
	for name in source_dirs.keys():
		files = source_dirs[name].glob('**/*.JPEG')
		for f in tqdm(list(files)):
			try:
				im = io.imread(str(f))
			except ValueError:
				skip_count += 1
				continue 

			try:
				tif_im = color.rgb2lab(im/255)
			except ValueError:

				### This added step skips 0 images in the dataset
				### Is this what we want?
				try:
					im = color.gray2rgb(im)
					tif_im = color.rgb2lab(im/255)
				except ValueError:
					skip_count += 1
					continue

				### Else comment above and do
				# skip_count += 1
				# continue

			save_path = get_save_path(name, f, root)
			io.imsave(str(save_path), tif_im)
	print(skip_count)

def get_save_path(name, f, root):
	words = list(f.parts[f.parts.index(name):])
	words[0], words[-1]= f'{name}_tif', words[-1].replace('.JPEG', '.TIF')
	words = ['..', '..', 'dataset'] + words
	save_path = root.joinpath(*words)
	save_path_dir = root.joinpath(*words[:-1])
	Path.mkdir(save_path_dir, parents=True, exist_ok=True)
	return save_path

if __name__ == '__main__':
	convert_to_tif()
