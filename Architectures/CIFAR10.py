import torch as t
import torch.nn as nn

class Generator(nn.Module):
	"""
		Selected Generator for CIFAR10 dataset. This produces images of size 32 x 32.
		Use this module for any GANs to be trained on CIFAR10.
	"""
	def __init__(self, n_z, ngpu, gen_type):
		"""
		Function to construct a Generator instance
		Args:
			n_z		: Dimensionality of the noise
			ngpu		: Number of GPUs to be used
			gen_type	: Type of Generator: 
						'conv' for normal convolutional,
						'resi' for residual blocks
		"""
		super(Generator, self).__init__()

		assert ngpu >= 0, "Number of GPUs has to be non-negative"
		assert n_z > 0, "Dimensionality of the noise vector has to be positive"
		
		if gen_type == 'conv':
			# Architecture: Specified as follows:
			# |   INPUT 	---->	  OUTPUT	(	   ACTIVATIONS 		  ) |
			# |    n_z	---->	   4096		(BATCHNORM_1D, LEAKY_RELU, DROPOUT) |
			# |  4x4x256	---->    8X8X128	(       BATCHNORM_2D, RELU 	  ) |
			# |  8x8x128	---->    16x16x64	(       BATCHNORM_2D, RELU	  ) |
			# |  16X16X64	---->	 32X32X3	(       BATCHNORM_2D, TANH	  ) |
			dropout_prob	= 0.4
			leaky_coeff	= 0.2
			
			# Fully Connected Section
			layer	= 1
			main	= nn.Sequential()
			main.add_module('Linear_{0}-{1}-{2}'.format(layer, n_z, 4096), nn.Linear(n_z, 4096))
			main.add_module('BatchNorm1d_{0}-{1}'.format(layer, 4096), nn.BatchNorm1d(4096))
			main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(leaky_coeff, inplace=True))
			main.add_module('Dropout_{0}'.format(layer), nn.Dropout(dropout_prob))
			self.fc	= main
			
			# Convolutional Section
			layer	= layer + 1
			main	= nn.Sequential()
			main.add_module('ConvTranspose2d_{0}-{1}-{2}'.format(layer, 256, 128),
					nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1))
			main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 128), nn.BatchNorm2d(128))
			main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
		
			layer	= layer + 1
			main.add_module('ConvTranspose2d_{0}-{1}-{2}'.format(layer, 128, 64),
					nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1))
			main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 64), nn.BatchNorm2d(64))
			main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
			
			layer	= layer + 1
			main.add_module('ConvTranspose2d_{0}-{1}-{2}'.format(layer, 64, 3),
					nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1))
			main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 64), nn.BatchNorm2d(3))
			main.add_module('TanH_{0}'.format(layer), nn.Tanh())
			self.cv	= main
		
		elif gen_type == 'resi':
			raise NotImplementedError
			
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
		Selected Discriminator for CIFAR10 dataset. This discriminates images of size 32 x 32.
		Use this module for any GANs to be trained on CIFAR10.
	"""
	def __init__(self, ngpu, dis_type):
		"""
		Function to construct a Discriminator instance
		Args:
			ngpu		: Number of GPUs to be used
			dis_type	: Type of Discriminator:
						'conv' for normal convolutional,
						'resi' for residual blocks
		"""
		super(Discriminator, self).__init__()

		assert ngpu >= 0, "Number of GPUs has to be non-negative"

		if dis_type == 'conv':
			# Architecture: Specified as follows: 
			# |   INPUT 	---->	  OUTPUT	(	   ACTIVATIONS 		  ) |
			# |  32x32x3	---->	 16x16x64	(BATCHNORM_2D, LEAKY_RELU, DROPOUT) |
			# |  16X16X64	---->    8X8X128	(BATCHNORM_2D, LEAKY_RELU, DROPOUT) |
			# |  8x8x128	---->    4x4x256	(      BATCHNORM_2D, LEAKY_RELU	  ) |
			# |   4096	---->	    1		(     	    SIGMOID		  ) |
			dropout_prob	= 0.4
			leaky_coeff	= 0.2

			# Convolution Section
			layer	= 1
			main	= nn.Sequential()
			main.add_module('Conv2d_{0}-{1}-{2}'.format(layer, 3, 64), 
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1))
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
					nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1))
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

		elif dis_type == 'resi':
			raise NotImplementedError
			
		
	def forward(self, input):
		input	= input.view(-1, 3, 32, 32)
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
