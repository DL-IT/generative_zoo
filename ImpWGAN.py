import torch as t
import utilities as u
import torch.nn as nn
import torch.utils.data as d_utils
import torchvision.utils as tv_utils
from torch.autograd import Variable as V

class ImpWGAN(object):
	def __init__(self, image_size, n_z, n_chan, hiddens, ngpu):
		"""
		ImpWGAN object. This class is a wrapper of a generalized improved WGAN as explained in the paper:
		IMPROVED TRAINING OF WASSERSTEIN GANs by Gulrajani et.al.
		
		Instance of this class initializes the Generator and Discriminator.
		Arguments:
			image_size		= Height / width of the real images
			n_z			= Dimensionality of the latent space
			n_chan			= Number of channels of the real images
			hiddens			= Number of feature maps in the first layer of the generator and discriminator
						  Format:
						  	hiddens = {'gen': n_gen_hidden, 
						  		   'dis': n_dis_hidden
						  		  }
			ngpu			= Number of gpus to be allocated, if to be run on gpu
		"""
		super(ImpWGAN, self).__init__()
		self.Gen_net	= Generator(image_size, n_z, n_chan, hiddens['gen'], ngpu)
		self.Dis_net	= Discriminator(image_size, n_chan, hiddens['dis'], ngpu)
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
		optimizer_details['gen']['params']	= self.Gen_net.parameters()
		optimizer_details['dis']['params']	= self.Dis_net.parameters()
		G_optmzr	= u.get_optimizer_with_params(optimizer_details['gen'])
		D_optmzr	= u.get_optimizer_with_params(optimizer_details['dis'])
		
		inpt	= t.FloatTensor(batch_size, self.n_chan, self.image_size, self.image_size)
		noise	= t.FloatTensor(batch_size, self.n_z)
		pos	= t.FloatTensor([1])
		neg	= pos.mul(-1)
		
		if display_images == True:
			fixed_noise	= t.randn(batch_size, self.n_z, 1, 1)
			
		if 'init_scheme' is misc_options:
			self.Gen_net.apply(u.weight_init_scheme)
			self.Dis_net.apply(u.weight_init_scheme)
			
		if self.ngpu > 0:
			inpt	= inpt.cuda()
			noise	= noise.cuda()
			pos	= pos.cuda()
			neg	= neg.cuda()
			if display_images == True:
				fixed_noise	= fixed_noise.cuda()
				
			self.Gen_net	= self.Gen_net.cuda()
			self.Dis_net	= self.Dis_net.cuda()
			
		d_loader	= d_utils.DataLoader(dataset, batch_size, shuffle=True)
		
		# Train loop
		# Details to be followed:
		# 1. Train the discriminator first for dis_iters_per_gen_iter times. Train the discriminator with reals, then fakes and then evaluate the gradient penalty term also
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
					self.Dis_net.zero_grad()
					cur_data	= d_iter.next()
					i		= i + 1
					
					# Training with reals. These are obviously True in the discriminator's POV
					X, _	= cur_data
					if inpt.size() != X.size():
						inpt.resize_(X.size(0), X.size(1), X.size(2), X.size(3))
					inpt.copy_(X)
					inptV	= V(inpt)
					
					otpt	= self.Dis_net(inptV)
					err_D_r	= (otpt.mean(0)).view(1)
					err_D_r.backward(neg)
					
					# Training with fakes. These are false in the discriminator's POV
					
					# We want same amount of fake data as real data
					if noise.size(0) != inpt.size(0):
						noise.resize_(inpt.size(0), noise.size(1), noise.size(2), noise.size(3))
					noise.normal_(0, 1)
					noiseV	= V(noise)
					X_f	= self.Gen_net(noiseV)
					otpt	= self.Dis_net(X_f)
					err_D_f	= (otpt.mean(0)).view(1)
					err_D_f.backward(pos)
								
					# Gradient penalty term
					grad_pen	= self.calc_grad_pen(X, X_f.cpu().data, lmbda)
					grad_pen.backward(pos)
					
					err_D	= err_D_r - err_D_f
					D_optmzr.step()
					j	= j + 1
					
				# Training the generator
				# We don't want to evaluate the gradients for the Discriminator during Generator training
				for params in self.Dis_net.parameters():
					params.requires_grad	= False
					
				for params in self.Gen_net.parameters():
					params.requires_grad	= True
					
				self.Gen_net.zero_grad()
				# The fake are real in the Generator's POV
				noise.normal_(0, 1)
				noiseV	= V(noise)
				X_gen	= self.Gen_net(noiseV)
				otpt	= self.Dis_net(X_gen)
				err_G	= (otpt.mean(0)).view(1)
				err_G.backward(neg)
				G_optmzr.step()
				
				gen_iters	= gen_iters + 1
				
				# Showing the Progress every show_period iterations
				if gen_iters % show_period == 0:
					print('[{0}/{1}]\tDiscriminator Error:\t{2}\tGenerator Error:\t{3}'.format(gen_iters, n_iters, round(err_D.data[0], 5), round(err_G.data[0], 5)))

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
			t.save(self.Gen_net.state_dict(), 'ImpWGAN_Gen_net_trained_model.pth')
			t.save(self.Dis_net.state_dict(), 'ImpWGAN_Dis_net_trained_model.pth')
			print('Training over and model(s) saved')

		elif flag == True:
			print('Training is over')
			
	def calc_grad_pen(self, real_data, fake_data, lmbda):
	
		epsilon	= t.FloatTensor(real_data.size(1), real_data.size(2), real_data.size(3)).uniform_(0, 1)
		epsilon = epsilon.expand_as(real_data)
					
		# The interpolate between real and fake is X_hat.
		X_hat	= epsilon*real_data + (1 - epsilon)*fake_data
		if self.ngpu > 0:
			X_hat	= X_hat.cuda()
		X_hatV	= V(X_hat, requires_grad=True)
		otpt	= self.Dis_net(X_hatV)
					
		gradients	= t.autograd.grad(outputs=otpt.mean(0).view(1), inputs=X_hatV, create_graph=True, retain_graph=True, only_inputs=True)[0]
		grad_pen	= (gradients.norm(2, dim=1) - 1).pow(2).mean().mul(lmbda)
		
		return grad_pen

class Generator(nn.Module):
	def __init__(self, image_size, n_z, n_chan, n_hidden, ngpu):
		super(Generator, self).__init__()
		
		assert image_size % 16 == 0, "Image size should be a multiple of 16"
		
		self.n_z	= n_z
		self.n_hidden	= n_hidden
		self.n_chan	= n_chan
		self.ngpu	= ngpu
		self.image_size	= image_size

		layer	= 1
		main_fc	= nn.Sequential()
		
		# Details to be followed
		# 1. First layer is a fully connected net
		# 2. Apply Transposed convolution until you reach image_size // 2
		# 3. Only last layer: tanh(). All others get ReLU()
		
		main_fc.add_module('linear_{0}-{1}-{2}'.format(layer, n_z, 4*4*n_hidden), nn.Linear(n_z, 4*4*n_hidden))
		main_fc.add_module('batchnorm_{0}-{1}'.format(layer, 4*4*n_hidden), nn.BatchNorm1d(4*4*n_hidden))
		main_fc.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
		
		# Current feature map size is 4 x 4
		cur_size	= 4

		main_conv	= nn.Sequential()
		while cur_size < image_size//2:
			layer	= layer + 1
			main_conv.add_module('conv_{0}-{1}-{2}'.format(layer, n_hidden, n_hidden//2), nn.ConvTranspose2d(n_hidden, n_hidden//2, kernel_size=4, stride=2, padding=1, bias=False))
			main_conv.add_module('batchnorm_{0}-{1}'.format(layer, n_hidden//2), nn.BatchNorm2d(n_hidden//2))
			main_conv.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
			cur_size	= cur_size * 2
			n_hidden	= n_hidden // 2
		
		# The last conv layer makes your generated image
		
		layer	= layer + 1
		main_conv.add_module('conv_{0}-{1}-{2}'.format(layer, n_hidden, n_chan), nn.ConvTranspose2d(n_hidden, n_chan, kernel_size=4, stride=2, padding=1, bias=False))
		main_conv.add_module('TanH_{0}'.format(layer), nn.Tanh())
		
		self.main_fc	= main_fc
		self.main_conv	= main_conv		
				
	def forward(self, input):
		if self.ngpu > 1:
			pass_	= nn.parallel.data_parallel(self.main_fc, input, range(0, self.ngpu))
			pass_	= pass_.view(-1, self.n_hidden, 4, 4)
			pass_	= nn.parallel.data_parallel(self.main_conv, pass_, range(0, self.ngpu))
		else:	
			pass_	= self.main_fc(input)
			pass_	= pass_.view(-1, self.n_hidden, 4, 4)
			pass_	= self.main_conv(pass_)
		return pass_
		
class Discriminator(nn.Module):
	def __init__(self, image_size, n_chan, n_hidden, ngpu):
		super(Discriminator, self).__init__()
		
		assert image_size % 16 == 0, "Image size should be a multiple of 16"
		
		layer		= 1
		main_conv	= nn.Sequential()
		
		# Details to be followed:
		# 1. LeakyReLU activations everywhere
		# 2. No BatchNorm anywhere
		# 3. Last Layer is a simple fully-connected layer
		
		main_conv.add_module('conv_{0}-{1}-{2}'.format(layer, n_chan, n_hidden), nn.Conv2d(n_chan, n_hidden, kernel_size=4, stride=2, padding=1, bias=False))
		main_conv.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(0.2, inplace=True))
		
		# Current feature map size is image_size/2 x image_size/2
		
		cur_size	= image_size // 2
		while cur_size > 4:
			layer	= layer + 1
			main_conv.add_module('conv_{0}-{1}-{2}'.format(layer, n_hidden, n_hidden*2), nn.Conv2d(n_hidden, n_hidden*2, kernel_size=4, stride=2, padding=1, bias=False))
			main_conv.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(0.2, inplace=True))
			cur_size	= cur_size // 2
			n_hidden	= n_hidden * 2
			
		# The last layer is a fully connected net
		main_fc	= nn.Sequential()
		
		layer	= layer + 1
		main_fc.add_module('linear_{0}-{1}-{2}'.format(layer, n_hidden*4*4, 1), nn.Linear(n_hidden*4*4, 1))
		
		self.main_fc	= main_fc
		self.main_conv	= main_conv
		self.n_hidden	= n_hidden
		self.n_chan	= n_chan
		self.ngpu	= ngpu

	def forward(self, input):
		if self.ngpu > 1:
			pass_	= nn.parallel.data_parallel(self.main_conv, input, range(0, self.ngpu))
			pass_.view(-1, 4*4*self.n_hidden)
			pass_	= self.main_fc(pass_)
		else:
			pass_	= self.main_conv(input)
			pass_	= pass_.view(-1, 4*4*self.n_hidden)
			pass_	= self.main_fc(pass_)
		return pass_
