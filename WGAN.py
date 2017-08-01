import torch as t
import DCGAN as dc
import MLPGAN as mlp
import utilities as u
import torch.nn as nn
import torch.utils.data as d_utils
import torchvision.utils as tv_utils
from torch.autograd import Variable as V

class WGAN(object):
	def __init__(self, image_size, n_z, n_chan, options, ngpu):
		"""
		WGAN object. This class is a wrapper of a generalized WGAN as explained in the paper:
		WASSERSTEIN GANS by Arjovsky et.al.
		
		Instance of this class initializes the Generator and the Discriminator.
		Arguments:
			image_size		= Height / width of the real images
			n_z			= Dimensionality of the latent space
			n_chan			= Number of channels of the real images
			options			= This specifies the type of network for the generator and discriminator, along with their parameters
						  Format:
						  	options = {'generator':
						  				{'type'	: dc/mlp
						  				 'hidden': common for both dc and mlp (meanings differ as mentioned in their respective classes
						  				 'depth': if mlp, the depth of the network
						  				},
						  				
						  		   'discriminator':
						  		   		<SAME AS ABOVE>
						  		   }
			ngpu			= Number of gpus to be allocated, if to be run of gpu
		"""
		super(WGAN, self).__init__()
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
		
	def train(self, dataset, batch_size, n_iters, dis_iters_per_gen_iter, clamps, optimizer_details, show_period=50, display_images=True, misc_options=['init_scheme', 'save_model']):
		"""
		Train function of the WGAN class. This starts training the model.
		Arguments:
			dataset			= Dataset object as from torchvision loader
			batch_size		= batch size to be used throughout the training
			n_iters			= Number of generator iterations to run the training for
			dis_iters_per_gen_iter	= Number of discriminator iterations per generator iteration
			clamps			= The weight thresholding clamps. 
						  Format:
						  	clamps	= {'lower': <lower bound>,
						  		   'upper': <upper bound>
						  		  }
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
			misc_options	(opt)	= List of strings. 
						  Add 'init_scheme' to the list, if you want to implement specific initialization schemes.
						  Add 'save_model' to the list, if you want to save the model after n_iters iterations of training
		"""
		optimizer_details['gen']['params'] 	= self.Gen_net.parameters()
		optimizer_details['dis']['params']	= self.Dis_net.parameters()
		G_optmzr	= u.get_optimizer_with_params(optimizer_details['gen'])
		D_optmzr	= u.get_optimizer_with_params(optimizer_details['dis'])
		
		inpt	= t.FloatTensor(batch_size, self.n_chan, self.image_size, self.image_size)
		noise	= t.FloatTensor(batch_size, self.n_z, 1, 1)
		
		if display_images == True:
			fixed_noise	= t.randn(batch_size, self.n_z, 1, 1)
			
		if 'init_scheme' in misc_options:
			self.Gen_net.apply(u.weight_init_scheme)
			self.Dis_net.apply(u.weight_init_scheme)
			
		if self.ngpu > 0:
			inpt	= inpt.cuda()
			noise	= noise.cuda()
			if display_images == True:
				fixed_noise	= fixed_noise.cuda()
				
			self.Gen_net	= self.Gen_net.cuda()
			self.Dis_net	= self.Dis_net.cuda()
			
		d_loader	= d_utils.DataLoader(dataset, batch_size, shuffle=True)
		
		# Train loop
		# Details to be followed:
		# 1. Train the discriminator first for dis_iters_per_gen_iter times. Train the discriminator with reals and then with fakes
		# 2. Train the generator after training the discriminator
		
		gen_iters	= 0
		flag		= False
		print('Training has started')
		while not flag:
			d_iter	= iter(d_loader)
			i	= 0
			while i < len(d_loader):
			
				# Training the discriminator
				# We don't want to evaluate the gradients for the Generator during Discriminator training
				for params in self.Gen_net.parameters():
					params.requires_grad	= False
				
				for params in self.Dis_net.parameters():
					params.requires_grad	= True
					
				j	= 0
				# Train the discriminator dis_iters_per_gen_iter times
				while j < dis_iters_per_gen_iter and i < len(d_loader):
					for params in self.Dis_net.parameters():
						params.data.clamp_(clamps['lower'], clamps['upper'])
						
					self.Dis_net.zero_grad()
					cur_data	= d_iter.next()
					i		= i + 1
					
					# Training with reals. These are obviously true in the discriminator's POV
					X, _	= cur_data
					if inpt.size() != X.size():
						inpt.resize_(X.size(0), X.size(1), X.size(2), X.size(3))
					inpt.copy_(X)
					inptV	= V(inpt)
					
					otpt	= self.Dis_net(inptV)
					otpt	= u.de_sigmoid(otpt)
					err_D_r	= (otpt.mean(0)).view(1)
					
					# Training with fakes. These are false in the discriminator's POV
					
					# We want same amount of fake data as real data
					if noise.size(0) != inpt.size(0):
						noise.resize_(inpt.size(0), noise.size(1), noise.size(2), noise.size(3))
					noise.normal_(0, 1)
					noiseV	= V(noise)
					X_f	= self.Gen_net(noiseV)
					otpt	= self.Dis_net(X_f)
					otpt	= u.de_sigmoid(otpt)
					err_D_f	= (otpt.mean(0)).view(1)
					
					err_D	= err_D_r - err_D_f
					err_D.backward()
					D_optmzr.step()
					j	= j + 1
					
				# Training the generator
				# We don't want to evaluate the gradients for the Discriminator during Generator training
				for params in self.Dis_net.parameters():
					params.requires_grad	= False
				
				for params in self.Gen_net.parameters():
					params.requires_grad	= True
					
				self.Gen_net.zero_grad()
				# The fake are reals in the Generator's POV
				noise.normal_(0, 1)
				noiseV	= V(noise)
				X_gen	= self.Gen_net(noiseV)
				otpt	= self.Dis_net(X_gen)
				otpt	= u.de_sigmoid(otpt)
				err_G	= (otpt.mean(0)).view(1)
				err_G.backward()
				G_optmzr.step()
				
				gen_iters	= gen_iters + 1

				# Showing the Progress every show_period iterations
				if gen_iters % show_period == 0:
					print('[{0}/{1}]\tDiscriminator Error:\t{2}\tGenerator Error:\t{3}'.format(gen_iters, n_iters, err_D.data[0], err_G.data[0]))
					
				# Saving the generated images every show_period*5 iterations
				if display_images == True:
					if gen_iters % (show_period*5) == 0:
						gen_imgs	= self.Gen_net(V(fixed_noise))
						
						# Normalizing the images to look better
						if self.n_chan > 1:
							gen_imgs.data	= gen_imgs.data.mul(0.5).add(0.5)
						tv_utils.save_image(gen_imgs.data, 'WGAN_Generated_images@iteration={0}.png'.format(gen_iters))

				if gen_iters == n_iters:
					flag	= True
					break
				
		if 'save_model' in misc_options and flag == True:
			t.save(self.Gen_net.state_dict(), 'WGAN_Gen_net_trained_model.pth')
			t.save(self.Dis_net.state_dict(), 'WGAN_Dis_net_trained_model.pth')
			print('Training over and model(s) saved')

		elif flag == True:
			print('Training is over')
