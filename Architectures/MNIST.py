import torch as t
import torch.nn as nn

class Generator(nn.Module):
	"""
		Selected Generator for MNIST dataset. This produces images of size 28 x 28.
		Use this module for any GANs to be trained on MNIST.
	"""
	def __init__(self, n_z, ngpu):
		"""
		Function to construct a Generator instance
		Args:
			n_z		: Dimensionality of the noise
			ngpu		: Number of GPUs to be used
		
		"""
		super(Generator, self).__init__()

		assert ngpu >= 0, "Number of GPUs has to be non-negative"
		assert n_z > 0, "Dimensionality of the noise vector has to be positive"

		# Architecture: Specified as follows: 
		# |   INPUT 	---->	  OUTPUT	(	   ACTIVATIONS 		  ) |
		# |    n_z	---->	   4096		(BATCHNORM_1D, LEAKY_RELU, DROPOUT) |
		# |  4X4X256	---->    7X7X128	(       BATCHNORM_2D, RELU 	  ) |
		# |  7x7x128	---->    14x14x64	(       BATCHNORM_2D, RELU	  ) |
		# |  14X14X64	---->	 28X28X1	(       BATCHNORM_2D, TANH	  ) |
		dropout_prob	= 0.4
		leaky_coeff	= 0.2
		
		# Fully connected Section
		layer	= 1
		main	= nn.Sequential()
		main.add_module('Linear_{0}-{1}-{2}'.format(layer, n_z, 4096), nn.Linear(n_z, 4096))
		main.add_module('BatchNorm1d_{0}-{1}'.format(layer, 4096), nn.BatchNorm1d(4096))
		main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(leaky_coeff, inplace=True))
		main.add_module('Dropout_{0}'.format(layer), nn.Dropout(dropout_prob))
		self.fc	= main
		
		# Convolution Section
		layer	= layer + 1
		main	= nn.Sequential()
		main.add_module('ConvTranspose2d_{0}-{1}-{2}'.format(layer, 256, 128), 
				nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=0))
		main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 128), nn.BatchNorm2d(128))
		main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
		
		layer	= layer + 1
		main.add_module('ConvTranspose2d_{0}-{1}-{2}'.format(layer, 128, 64),
				nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1))
		main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 64), nn.BatchNorm2d(64))
		main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
		
		layer	= layer + 1
		main.add_module('ConvTranspose2d_{0}-{1}-{2}'.format(layer, 64, 1),
				nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1))
		main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 1), nn.BatchNorm2d(1))
		main.add_module('TanH_{0}'.format(layer), nn.Tanh())
		self.cv	= main
		
		self.n_z	= n_z
		self.ngpu	= ngpu
	
	def forward(self, input):
		input	= input.view(-1, self.n_z)
		if self.ngpu > 1:
			pass_	= nn.parallel.data_parallel(self.fc, input, range(0, self.ngpu))
			pass_	= pass_.view(-1, 256, 4, 4)
			pass_	= nn.parallel.data_parallel(self.cv, pass_, range(0, self.ngpu))
			return pass_
		else:
			pass_	= self.fc(input)
			pass_	= pass_.view(-1, 256, 4, 4)
			pass_	= self.cv(pass_)
			return pass_
			
class Discriminator(nn.Module):
	"""
		Selected Discriminator for MNIST dataset. This discriminates images of size 28 x 28.
		Use this module for any GANs to be trained on MNIST.
	"""
	def __init__(self, ngpu):
		"""
		Function to construct a Discriminator instance
		Args:
			ngpu		: Number of GPUs to be used
		
		"""
		super(Discriminator, self).__init__()

		assert ngpu >= 0, "Number of GPUs has to be non-negative"
		
		# Architecture: Specified as follows: 
		# |   INPUT 	---->	  OUTPUT	(	   ACTIVATIONS 		  ) |
		# |  28x28x1	---->	 14x14x64	(BATCHNORM_2D, LEAKY_RELU, DROPOUT) |
		# |  14X14X64	---->    7X7X128	(BATCHNORM_2D, LEAKY_RELU, DROPOUT) |
		# |  7x7x128	---->    4x4x256	(      BATCHNORM_2D, LEAKY_RELU	  ) |
		# |   4096	---->	    1		(     	    SIGMOID		  ) |
		dropout_prob	= 0.4
		leaky_coeff	= 0.2
		
		# Convolution Section
		layer	= 1
		main	= nn.Sequential()
		main.add_module('Conv2d_{0}-{1}-{2}'.format(layer, 1, 64), 
				nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1))
		main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 64), nn.BatchNorm2d(64))
		main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(leaky_coeff, inplace=True))
		main.add_module('Dropout_{0}'.format(layer), nn.Dropout(dropout_prob))
		
		layer	= layer + 1
		main.add_module('Conv2d_{0}-{1}-{2}'.format(layer, 64, 128),
				nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1))
		main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 128), nn.BatchNorm2d(128))
		main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(leaky_coeff, inplace=True))
		main.add_module('Dropout_{0}'.format(layer), nn.Dropout(dropout_prob))
		
		layer	= layer + 1
		main.add_module('Conv2d_{0}-{1}-{2}'.format(layer, 128, 256),
				nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=0))
		main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 256), nn.BatchNorm2d(256))
		main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(leaky_coeff, inplace=True))
		self.cv	= main
		
		# Fully connected Section
		layer	= layer + 1
		main	= nn.Sequential()
		main.add_module('Linear_{0}-{1}-{2}'.format(layer, 4096, 1), nn.Linear(4096, 1))
		main.add_module('Sigmoid_{0}'.format(layer), nn.Sigmoid())
		self.fc	= main
		
		self.ngpu	= ngpu
		
	def forward(self, input):
		input	= input.view(-1, 1, 28, 28)
		if self.ngpu > 1:
			pass_	= nn.parallel.data_parallel(self.cv, input, range(0, self.ngpu))
			pass_	= pass_.view(-1, 4096)
			pass_	= nn.parallel.data_parallel(self.fc, pass_, range(0, self.ngpu))
			return pass_
		else:
			pass_	= self.cv(input)
			pass_	= pass_.view(-1, 4096)
			pass_	= self.fc(pass_)
			return pass_
