import torch
from pathlib import Path
from PIL import Image
import numpy as np

class Dataset(torch.utils.data.Dataset):
	def __init__(self, dataset):
		self.dataset = dataset

	def __getitem__(self, i):
		im = Image.open(self.dataset[i])
		im = np.array(im)
		X, y = im[:,:,0], im[:,:,1:]
		return X, y
		
	def __len__(self):
		return len(self.dataset)

    
    
def prepare(set_spec):
    X = list(set_spec.glob('**/*.TIF'))

    dataset = Dataset(X)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=25, num_workers=0, shuffle=True)
    
    return train_loader

def return_loaders():
    paths = ['../dataset/test_tif', '../dataset/train_tif', '../dataset/val_tif']
    paths = list(map(Path, paths))

    dataset = {}
    dataset['test'], dataset['train'], dataset['validation'] = map(prepare, paths)
    
    return dataset
