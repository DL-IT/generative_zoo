import torch as t
import utilities as u
import torch.nn as nn
import torch.utils.data as d_utils
import torchvision.utils as tv_utils
from torch.autograd import Variable as V

class DCGAN(object):
	def __init__(self, arch, ngpu, loss='BCE'):
		"""
		DCGAN object. This class is a wrapper of a generalized DCGAN as explained in the paper: 
			UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS by Alec Radford et.al.
		
		Instance of this class initializes the Generator and the Discriminator.
		Arguments:
			arch			= Architecture to use:
							"CIFAR10" for CIFAR10 dataset
							"MNIST" for MNIST dataset
							"Generic" for a general Generator and Discriminator architecture
						  
						  For CIFAR10/MNIST: Need to input n_z also. 
						  For Generic: Need to input image_size, n_z, n_chan, hiddens
						  
						  {'arch_type': <arch_type>,
						   'params'   : <params> as above
						  }
						  
						  Example:
						  {'arch_type': "CIFAR10",
						   'params'   : {'n_z' : 128
						   		}
						  }
						  {'arch_type': "Generic",
						   'params'   : {'image_size'	: 32,
						    		 'n_z'		: 128,
						    		 'n_chan'	: 3,
						    		 'hiddens'	: <see below>
						    		}
						  }

							image_size		= Height / width of the real images
							n_z			= Dimensionality of the latent space
							n_chan			= Number of channels of the real images
							hiddens			= Number of feature maps in the first layer of the generator and discriminator
										  Format:
										  	hiddens = {'gen': n_gen_hidden, 
										  		   'dis': n_dis_hidden
										  		  }
			ngpu			= Number of gpus to be allocated, if to be run on gpu
			loss			= The loss function to be used
		"""
		super(DCGAN, self).__init__()
		if arch['arch_type'] == 'Generic':
			from Generic import Generator
			self.Gen_net	= Generator(
							image_size	= arch['params']['image_size'],
							n_z		= arch['params']['n_z'], 
							n_chan		= arch['params']['n_chan'],
							n_hidden	= arch['params']['hiddens']['gen'],
							ngpu		= ngpu
							
						   )
						   
			from Generic import Discriminator
			self.Dis_net	= Discriminator(
							image_size	= arch['params']['image_size'],
							n_chan		= arch['params']['n_chan'],
							n_hidden		= arch['params']['hiddens']['dis'],
							ngpu		= ngpu
							)

			self.image_size	= arch['params']['image_size']
			self.n_chan	= arch['params']['n_chan']
							
		elif arch['arch_type'] == 'MNIST':
			from MNIST import Generator
			self.Gen_net	= Generator(
							n_z		= arch['params']['n_z'],
						   	ngpu		= ngpu
						   )
						   
			from MNIST import Discriminator
			self.Dis_net	= Discriminator(
							ngpu		= ngpu
							)

			self.image_size	= 28
			self.n_chan	= 1
		
		elif arch['arch_type'] == 'CIFAR10':
			from CIFAR10 import Generator
			self.Gen_net	= Generator(
							n_z		= arch['params']['n_z'],
							ngpu		= ngpu,
							gen_type	= arch['params']['gen_type']
						   )
						   
			from CIFAR10 import Discriminator
			self.Dis_net	= Discriminator(
							ngpu		= ngpu,
							dis_type	= arch['params']['dis_type']
							)

			self.image_size	= 32
			self.n_chan	= 3
						
		self.ngpu	= ngpu
		self.n_z	= arch['params']['n_z']
		if loss == 'BCE':
			self.loss	= nn.BCELoss()
		elif loss == 'MSE':
			self.loss	= nn.MSELoss()  
		
	def train(self, dataset, batch_size, n_iters, optimizer_details, show_period=50, display_images=True, misc_options=['init_scheme', 'save_model']):
		"""
		Train function of the DCGAN class. This starts training the model.
		Arguments:
			dataset			= Dataset object as from torchvision loader
			batch_size		= batch size to be used throughout the training
			n_iters			= Number of generator iterations to run the training for
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
		optimizer_details['gen']['params']	= self.Gen_net.parameters()
		optimizer_details['dis']['params']	= self.Dis_net.parameters()
		G_optmzr	= u.get_optimizer_with_params(optimizer_details['gen'])
		D_optmzr	= u.get_optimizer_with_params(optimizer_details['dis'])

		inpt	= t.FloatTensor(batch_size, self.n_chan, self.image_size, self.image_size)
		noise	= t.FloatTensor(batch_size, self.n_z, 1, 1)
		label	= t.FloatTensor(batch_size)
		if display_images == True:
			fixed_noise	= t.randn(batch_size, self.n_z, 1, 1)
		
		if 'init_scheme' in misc_options:
			self.Gen_net.apply(u.weight_init_scheme)
			self.Dis_net.apply(u.weight_init_scheme)
		
		if self.ngpu > 0:
			inpt	= inpt.cuda()
			noise	= noise.cuda()
			label	= label.cuda()
			if display_images == True:
				fixed_noise	= fixed_noise.cuda()

			self.Gen_net	= self.Gen_net.cuda()
			self.Dis_net	= self.Dis_net.cuda()
			
		d_loader	= d_utils.DataLoader(dataset, batch_size, shuffle=True)
		
		# Train loop
		# Details to be followed:
		# 1. Train the discriminator first. Train the discriminator with reals and then with fakes
		# 2. Train the generator after training the discriminator.
		
		gen_iters	= 0
		flag		= False
		print('Training has started')
		while not flag:
			for i, itr in enumerate(d_loader):
				
				self.Gen_net.train()
				self.Dis_net.train()
				# Training the discriminator
				# We don't want to evaluate the gradients for the Generator during Discriminator training
				self.Dis_net.zero_grad()
				# Training with reals. These are obviously true in the discriminator's POV
				X, _	= itr
				if inpt.size() != X.size():
					inpt.resize_(X.size(0), X.size(1), X.size(2), X.size(3))
					label.resize_(X.size(0))
				inpt.copy_(X)
				label.fill_(1)
				
				inptV	= V(inpt)
				labelV	= V(label)
				
				otpt	= self.Dis_net(inptV)
				err_D_r	= self.loss(otpt, labelV)
				err_D_r.backward()
				
				# Training with fakes. These are false in the discriminator's POV
					
				# We want same amount of fake data as real data
				if noise.size(0) != inpt.size(0):
					noise.resize_(inpt.size(0), noise.size(1), noise.size(2), noise.size(3))
					label.resize_(X.size(0))
				noise.normal_(0, 1)
				label.fill_(0)
				
				noiseV	= V(noise)
				labelV	= V(label)
				
				X_f	= self.Gen_net(noiseV)
				otpt	= self.Dis_net(X_f.detach())
				err_D_f	= self.loss(otpt, labelV)
				err_D_f.backward()
				err_D	= err_D_r + err_D_f
				D_optmzr.step()
				
				# Training the generator
				# We don't want to evaluate the gradients for the Discriminator during Generator training
					
				self.Gen_net.zero_grad()
				# The fake are reals in the Generator's POV
				label.fill_(1)
				labelV	= V(label)
				
				otpt	= self.Dis_net(X_f)
				err_G	= self.loss(otpt, labelV)
				err_G.backward()
				G_optmzr.step()
				
				gen_iters	= gen_iters + 1
				
				# Showing the Progress every show_period iterations
				if gen_iters % show_period == 0:
					print('[{0}/{1}]\tDiscriminator Error:\t{2}\tGenerator Error:\t{3}'.format(gen_iters, n_iters, round(err_D.data[0], 5), round(err_G.data[0], 5)))
					
				# Saving the generated images every show_period*5 iterations
				if display_images == True:
					if gen_iters % (show_period*5) == 0:
						self.Gen_net.eval()
						gen_imgs	= self.Gen_net(V(fixed_noise))
						
						gen_imgs.data	= gen_imgs.data.mul(0.5).add(0.5)
						tv_utils.save_image(gen_imgs.data, 'DCGAN_Generated_images@iteration={0}.png'.format(gen_iters))

				if gen_iters == n_iters:
					flag	= True
					break
					
		if 'save_model' in misc_options and flag == True:
			t.save(self.Gen_net.state_dict(), 'DCGAN_Gen_net_trained_model.pth')
			t.save(self.Dis_net.state_dict(), 'DCGAN_Dis_net_trained_model.pth')
			print('Training over and model(s) saved')
			
		elif flag == True:
			print('Training is over')
