from re import L
import numpy as np
import os
from pathlib import Path
from numpy.lib.npyio import load
import warnings
from json import JSONEncoder
import json
import pprint

class NPYEncoder(JSONEncoder):
	def default(self, data):
		if isinstance(data, np.ndarray):
			return data.tolist()
		return JSONEncoder.default(self, data)


def _load_internal_memory():
	path = Path(__file__).parent.absolute() / 'npy.json'
	with open(path, 'r') as f:
		data = json.load(f)
	return _decode_json(data)

def _decode_json(dic):
	for key in dic.keys():
		dic[key] = np.array(dic[key])
	return dic

def _format(name):
	""" safe guard """
	if name.endswith('.npy'):
		return name.replace('.npy', '')
	else:
		return name

def _dump(data):
	path = Path(os.path.dirname(os.path.realpath(__file__))) / 'npy.json'
	with open(path, 'w') as f:
		json.dump(data, f, cls = NPYEncoder, sort_keys = True)

	path = Path(os.path.dirname(os.path.realpath(__file__))) / 'valid_keys.txt'
	with open(path, 'w') as f:
		for key in sorted(list_all()):
			f.write(f'{key}\n')

def _pretty_print(keys, as_string = False):
	prt = pprint.PrettyPrinter(indent = 1, depth = 2)
	string = '\n\n' + 'Valid keys = ' + prt.pformat(keys) + '\n'
	if as_string:
		return string
	else:
		print(string)


def load(name):
	""" load data from name """
	name = _format(name)

	if name not in list_all():
		string = _pretty_print(list_all(), as_string = True)
		raise ValueError(f'\nNo such file {name}. {string}')
	return memory[name]

def save(name, data):
	""" save data to file name """
	name = _format(name)
	memory[name] = data
	_dump(memory)


def rm(name):
	""" removes data and file """
	name = _format(name)
	if name in list_all():
		memory.pop(name)
		_dump(memory)
	else:
		string = _pretty_print(list_all(), as_string = True)
		warnings.warn(f'\n\nAction aborted no file named {name}. {string}')

def list_all(verbose = False):
	if verbose:
		_pretty_print(list(memory.keys()))
	return list(memory.keys())

memory = _load_internal_memory()

if __name__ == '__main__':
# 	test functions
	list_all(verbose = True)
	print(_format('test.npy'))
	save('test', np.zeros(shape = (100, 100)))
	save('test', np.zeros(shape = (100, 100)))
	list_all(verbose = True)
	print(load('test'))
	rm('test')
	list_all(verbose = True)
	load('authors_prior_probs.npy')
	# load('pippis_aventyr')
	rm('alla_barnen_i_bullerbyn')