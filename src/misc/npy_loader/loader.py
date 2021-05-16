from re import L
import numpy as np
import os
from pathlib import Path
from numpy.lib.npyio import load

def _load_internal_memory():
	""" private """
	dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
	files = dir_path.glob('*.npy')
	dic = {}
	for f in files:
		dic[f.name.replace('.npy', '')] = np.load(str(f))
	return dic

def load(name):
	return memory[name]

def save(name, data):
	dir_path = Path(__file__).parent.absolute() / f'{name}.npy' 
	memory[name] = data
	np.save(dir_path, data)

def list_all():
	print(memory.keys())

memory = _load_internal_memory()

if __name__ == '__main__':
	list_all()



	