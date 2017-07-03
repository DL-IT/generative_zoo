import torch as t
import numpy as np
import torch.nn as nn

# Generator net

class Generator(nn.Module):
	def __init__(self, image_size, n_z, n_chan, n_hidden, depth, ngpu):
		super(Generator, self).__init__()
		
		self.image_size	= image_size
		self.n_z	= n_z
		self.n_hidden	= n_hidden
		self.n_chan	= n_chan
		self.depth	= depth
		self.ngpu	= ngpu		

		layer	= 1
		main	= nn.Sequential()
		
		main.add_module('full_connect_{0}_{1}-{2}'.format(layer, n_z, n_hidden), nn.Linear(n_z, n_hidden))
		main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
		
		while layer < depth - 1:
			layer	= layer + 1
			main.add_module('full_connect_{0}_{1}-{2}'.format(layer, n_hidden, n_hidden), nn.Linear(n_hidden, n_hidden))
			main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
			
		layer		= layer + 1
		image_dim	= image_size*image_size*n_chan
		main.add_module('full_connect_{0}_{1}-{2}'.format(layer, n_hidden, image_dim), nn.Linear(n_hidden, image_dim))
		main.add_module('Tanh_{0}'.format(layer), nn.Tanh())

		self.main	= main
		
	def forward(self, input):
		if self.ngpu > 0:
			output	= nn.parallel.data_parallel(self.main, input, range(0, self.ngpu))
		else:
			output	= self.main(input)
		
		return output.view(-1, self.n_chan, self.image_size, self.image_size)
		
# Discriminator net

class Discriminator(nn.Module):
	def __init__(self, image_size, n_chan, n_hidden, depth, ngpu):
		super(Discriminator, self).__init__()
		
		self.image_size	= image_size
		self.n_chan	= n_chan
		self.n_hidden	= n_hidden
		self.depth	= depth
		self.ngpu	= ngpu
		
		layer		= 1
		image_dim	= image_size*image_size*n_chan
		main		= nn.Sequential()
		
		main.add_module('full_connect_{0}_{1}-{2}'.format(layer, image_dim, n_hidden), nn.Linear(image_dim, n_hidden))
		main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
		
		while layer < depth - 1:
			layer	= layer + 1
			main.add_module('full_connect_{0}_{1}-{2}'.format(layer, n_hidden, n_hidden), nn.Linear(n_hidden, n_hidden))
			main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
			
		layer		= layer + 1
		main.add_module('full_connect_{0}_{1}-{2}'.format(layer, n_hidden, 1), nn.Linear(n_hidden, 1))
		main.add_module('Sigmoid_{0}'.format(layer), nn.Sigmoid())
		
		self.main	= main
		
	def forward(self, input):
		input	= input.view(-1, self.n_chan*self.image_size*self.image_size)
		if self.ngpu > 0:
			output	= nn.parallel.data_parallel(self.main, input, range(0, self.ngpu))
		else:
			output	= self.main(input)
		
		return output.view(-1, 1)
		
def weight_init_scheme(mdl):
	mdl_name	= mdl.__class__.__name__
	if mdl_name.find('Linear') != -1:
		f_in	= mdl.weight.size(0)
		f_out	= mdl.weight.size(1)
		mdl.weight.data.uniform_(-np.sqrt(6/(f_in + f_out)), np.sqrt(6/(f_in + f_out)))
		mdl.bias.data.fill_(0)
