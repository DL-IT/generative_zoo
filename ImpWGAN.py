import torch as t
import DCGAN as dc
import MLPGAN as mlp
import utilities as u
import torch.nn as nn
import torch.utils.data as d_utils
import torchvision.utils as tv_utils
from torch.autograd import Variable as V

class ImpWGAN(object):
	def __init__(self, image_size, n_z, n_chan, options, ngpu):
		"""
		ImpWGAN object. This class is a wrapper of a generalized improved WGAN as explained in the paper:
		IMPROVED TRAINING OF WASSERSTEIN GANs by Gulrajani et.al.
		
		Instance of this class initializes the Generator and Discriminator.
		Arguments:
			image_size		= Height / width of the real images
			n_z			= Dimensionality of the latent space
			n_chan			= Number of channels of the real images
			options			= This specifies the type of the network for the generator and discriminator, along with their parameters
						  Format:
						  	options	= {'generator':
						  				{'type'	: dc/mlp
						  				 'hidden': common for both dc and mlp (meanings differ as mentioned in their respective classes
						  				 'depth' : if mlp, the depth of the network
						  				},
						  				
						  		   'discriminator':
						  		   		<SAME AS ABOVE>
						  		  }
			ngpu			= Number of gpus to be allocated, if to be run on gpu
		"""
		super(ImpWGAN, self).__init__()
		gen_type	= options['generator']['type']
		dis_type	= options['discriminator']['type']
		
		if gen_type == 'dc':
			self.Gen_net	= dc.Generator(image_size, n_z, n_chan, options['generator']['hidden'], ngpu)
		elif gen_type == 'mlp':
			self.Gen_net	= mlp.Generator(image_size, n_z, n_chan, options['generator']['hidden'], options['generator']['depth'], ngpu)
			
		if dis_type == 'dc':
			self.Dis_net	= dc.Discriminator(image_size, n_chan, options['discriminator']['hidden'], ngpu)
		elif dis_type == 'mlp':
			self.Dis_net	= mlp.Discriminator(image_size, n_chan, options['discriminator']['hidden'], options['discriminator']['depth'], ngpu)
			
		self.ngpu	= ngpu
		self.n_z	= n_z
		self.image_size	= image_size
		self.n_chan	= n_chan
		
	def train(self, dataset, batch_size, n_iters, dis_iters_per_gen_iter, lmbda, optimizer_details, show_period=50, display_images=True, misc_options=['init_scheme', 'save_model']):
		"""
		Train function of the ImpWGAN class. This starts training the model.
		Arguments:
			dataset			= Dataset object as from torchvision loader
			batch_size		= batch size to be used throughout the training
			n_iters			= Number of generator iterations to run the training for
			dis_iters_per_gen_iter	= Number of discriminator iterations per generator iteration
			lmbda			= The gradient penalty hyperparameter
			optimizer_details	= Dictionary representing the details for optimizers for generator and discriminator
						  Format:
						  	optimizer_details = {'gen':
						  					{'name'		: Name of optimizer,
						  					 'learn_rate'	: learning rate,
						  					 'betas'	: (beta_1, beta_2), 	=> Optional, if using Adam/Adamax
						  					 'momentum'	: momentum,		=> Optional, if using momentum SGD/NAG
						  					 'nesterov'	: True/False,		=> Optional, true if using NAG else otherwise
						  					},
						  			     'dis':
						  			     		<SAME AS ABOVE>
						  			     }
			show_period	(opt)	= Prints the errors with current iteration number every show_period iterations
			display_images	(opt)	= If true, saves the generated images from noise every show_period*5 iterations
			misc_options	(opt)	= List of strings
						  Add 'init_scheme' to the list, if you want to implement specific initialization schemes
						  Add 'save_model' to the list, if you want to save the model after n_iters iterations of training
		"""
		raise NotImplementedError
