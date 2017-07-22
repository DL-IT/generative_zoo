import torch as t
import utilities as u
import torch.nn as nn
import torch.utils.data as d_utils
import torchvision.utils as tv_utils
from torch.autograd import Variable as V

class DCGAN(object):
	def __init__(self, image_size, n_z, n_chan, hiddens, ngpu, loss='BCE'):
		"""
		DCGAN object. This class is a wrapper of a generalized DCGAN as explained in the paper: 
			UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS by Alec Radford et.al.
		
		Instance of this class initializes the Generator and the Discriminator.
		Arguments:
			image_size		= Height / width of the real images
			n_z			= Dimensionality of the latent space
			n_chan			= Number of channels of the real images
			hiddens			= Number of feature maps in the first layer of the generator and discriminator
						  Format:
						  	hiddens = {'gen': n_gen_hidden, 
						  		   'dis': n_dis_hidden
						  		  }
			ngpu			= Number of gpus to allocated, if to be run on gpu
			loss			= The loss function to be used
		"""
		super(DCGAN, self).__init__()
		self.Gen_net	= Generator(image_size, n_z, n_chan, hiddens['gen'], ngpu)
		self.Dis_net	= Discriminator(image_size, n_chan, hiddens['dis'], ngpu)
		self.ngpu	= ngpu
		self.n_z	= n_z
		self.image_size	= image_size
		self.n_chan	= n_chan
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
				
				# Training the discriminator
				# We don't want to evaluate the gradients for the Generator during Discriminator training
				for params in self.Gen_net.parameters():
					params.requires_grad	= False
				
				for params in self.Dis_net.parameters():
					params.requires_grad	= True

				self.Dis_net.zero_grad()
				# Training with reals. These are obviously true in the discriminator's POV
				X, _	= itr
				if inpt.size() != X.size():
					inpt.resize_as_(X)
				inpt.copy_(X)
				label.fill_(1)
				
				inptV	= V(inpt)
				labelV	= V(label)
				
				otpt	= self.Dis_net(inptV)
				err_D_r	= self.loss(otpt, labelV)
				
				# Training with fakes. These are false in the discriminator's POV
					
				# We want same amount of fake data as real data
				if noise.size(0) != inpt.size(0):
					noise.resize_(inpt.size(0), noise.size(1), noise.size(2), noise.size(3))
				noise.normal_(0, 1)
				label.fill_(0)
				
				noiseV	= V(noise)
				labelV	= V(label)
				
				X_f	= self.Gen_net(noiseV)
				otpt	= self.Dis_net(X_f)
				err_D_f	= self.loss(otpt, labelV)
				err_D	= err_D_r + err_D_f
				err_D.backward()
				D_optmzr.step()
				
				# Training the generator
				# We don't want to evaluate the gradients for the Discriminator during Generator training
				for params in self.Dis_net.parameters():
					params.requires_grad	= False
				
				for params in self.Gen_net.parameters():
					params.requires_grad	= True
					
				self.Gen_net.zero_grad()
				# The fake are reals in the Generator's POV
				noise.normal_(0, 1)
				label.fill_(1)
				
				noiseV	= V(noise)
				labelV	= V(label)
				
				X_gen	= self.Gen_net(noiseV)
				otpt	= self.Dis_net(X_gen)
				err_G	= self.loss(otpt, labelV)
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
						gen_imgs.data	= gen_imgs.data.mul(0.5).add(0.5)
						tv_utils.save_image(gen_imgs.data, 'Generated_images@iteration={0}.png'.format(gen_iters))

				if gen_iters == n_iters:
					flag	= True
					break
					
		if 'save_model' in misc_options and flag == True:
			torch.save(self.Gen_net.state_dict(), 'DCGAN_Gen_net_trained_model.pth')
			torch.save(self.Dis_net.state_dict(), 'DCGAN_Dis_net_trained_model.pth')
		print('Training over and model(s) saved')
				
# Generator net

class Generator(nn.Module):
	def __init__(self, image_size, n_z, n_chan, n_hidden, ngpu):
		super(Generator, self).__init__()
		
		assert image_size % 16 == 0, "Image size should be a multiple of 16"
		
		layer	= 1
		main	= nn.Sequential()
		
		# Details to be followed:
		# 1. ReLU for activation for all but the last layer
		# 2. Batchnorm for all but the last layer
		# 3. No fully connected layers
		
		# The first conv layer transforms the noise into a set of n_hidden feature maps

		main.add_module('conv_{0}-{1}-{2}'.format(layer, n_z, n_hidden), nn.ConvTranspose2d(n_z, n_hidden, kernel_size=4, stride=1, padding=0, bias=False))
		main.add_module('batchnorm_{0}-{1}'.format(layer, n_hidden), nn.BatchNorm2d(n_hidden))
		main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
		
		# Current feature map size is 4
		cur_size	= 4
		
		# Keep enlarging the feature map until before it reaches the size of the image
		while cur_size < image_size//2 :
			layer = layer + 1
			main.add_module('conv_{0}-{1}-{2}'.format(layer, n_hidden, n_hidden//2), nn.ConvTranspose2d(n_hidden, n_hidden//2, kernel_size=4, stride=2, padding=1, bias=False))
			main.add_module('batchnorm_{0}-{1}'.format(layer, n_hidden//2), nn.BatchNorm2d(n_hidden//2))
			main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
			
			n_hidden	= n_hidden // 2
			cur_size	= cur_size * 2
			
		# The last conv layer transforms existing feature maps into n_chan feature maps of the size of the image
		
		layer	= layer + 1
		main.add_module('conv_{0}-{1}-{2}'.format(layer, n_hidden, n_chan), nn.ConvTranspose2d(n_hidden, n_chan, kernel_size=4, stride=2, padding=1, bias=False))
		main.add_module('TanH_{0}'.format(layer), nn.Tanh())
		
		self.main	= main
		self.image_size	= image_size
		self.n_z	= n_z
		self.n_hidden	= n_hidden
		self.n_chan	= n_chan
		self.ngpu	= ngpu
		
	def forward(self, input):
		if self.ngpu > 0:
			output	= nn.parallel.data_parallel(self.main, input, range(0, self.ngpu))
		else:
			output	= self.main(input)			
		return output
		
# Discriminator net

class Discriminator(nn.Module):
	def __init__(self, image_size, n_chan, n_hidden, ngpu):
		super(Discriminator, self).__init__()
	
		assert image_size % 16 == 0, "Image size should be a multiple of 16"
	
		layer	= 1
		main	= nn.Sequential()
	
		# Details to be followed:
		# 1. Leaky ReLU activation for all but the first and last layer
		# 2. Batchnorm for all but the last layer
		# 3. No fully connected layers
	
		# The first conv layer transforms the image into a smaller feature map
	
		main.add_module('conv_{0}-{1}-{2}'.format(layer, n_chan, n_hidden), nn.Conv2d(n_chan, n_hidden, kernel_size=4, stride=2, padding=1, bias=False))
		main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(0.2, inplace=True))
		
		# Current feature map size is image_size/2
		
		cur_size	= image_size // 2
		
		# Keep diminishing the size of feature map until before it reaches 4
		while cur_size > 4:
			layer	= layer + 1
			main.add_module('conv_{0}-{1}-{2}'.format(layer, n_hidden, n_hidden*2), nn.Conv2d(n_hidden, n_hidden*2, kernel_size=4, stride=2, padding=1, bias=False))
			main.add_module('batchnorm_{0}-{1}'.format(layer, n_hidden*2), nn.BatchNorm2d(n_hidden*2))
			main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(0.2, inplace=True))
			
			cur_size	= cur_size // 2
			n_hidden	= n_hidden * 2
			
		# The last conv layer transforms the K x 4 x 4 feature map into a K dimensional output vector
		
		layer	= layer + 1		
		main.add_module('conv_{0}-{1}-{2}'.format(layer, n_hidden, 1), nn.Conv2d(n_hidden, 1, kernel_size=4, stride=1, padding=0, bias=False))
		main.add_module('Sigmoid_{0}'.format(layer), nn.Sigmoid())
		
		self.main	= main
		self.image_size	= image_size
		self.n_hidden	= n_hidden
		self.n_chan	= n_chan
		self.ngpu	= ngpu
		
	def forward(self, input):
		if self.ngpu > 0:
			output	= nn.parallel.data_parallel(self.main, input, range(0, self.ngpu))
		else:
			output	= self.main(input)
		return output.view(-1, 1)
