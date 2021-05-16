from re import L
import numpy as np
import os
from pathlib import Path
from numpy.lib.npyio import load

def _load_internal_memory():
	""" private """
	dir_path = Path(os.path.dirname(os.path.realpath(__file__))) / 'npy'
	files = dir_path.glob('*.npy')
	dic = {}
	for f in files:
		dic[f.name.replace('.npy', '')] = np.load(str(f))
	return dic

def load(name):
	if name not in list_all():
		string = ' : '.join(list_all())
		raise ValueError(f'\nNo such file {name}.\nAvailable files are:\n{string}')
	return memory[name]

def save(name, data):
	dir_path = Path(__file__).parent.absolute() / 'npy' / f'{name}.npy'
	np.save(str(dir_path), data)
	memory[name] = data

def rm(name):
	""" removes data and file """
	dir_path = Path(__file__).parent.absolute() / 'npy' / f'{name}.npy'
	dir_path.unlink()
	memory.pop(name)

def list_all(print = False):
	if print:
		print(memory.keys())
	return memory.keys()

memory = _load_internal_memory()

if __name__ == '__main__':
	# test functions
	list_all()
	save('test', np.zeros(shape = (100, 100)))
	list_all()
	print(load('test'))
	rm('test')
	



	