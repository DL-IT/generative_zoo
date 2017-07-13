import torch as t
import utilities as u
import torch.nn as nn
import torch.utils.data as d_utils
import torchvision.utils as tv_utils
from torch.autograd import Variable as V

t.manual_seed(29)

class ALPHAGAN(object):
	def __init__(self, image_size, n_z, n_chan, hiddens, code_dis_depth, ngpu):
		"""
		ALPHAGAN object. This class is a wrapper of a generalized ALPHA-GAN as explained in the paper:
			VARIATIONAL APPROACHES FOR AUTO-ENCODER GENERATIVE ADEVERSARIAL NETWORKS by Mihaela Rosca and Balaji Lakshminarayanan et.al.
			
		Instance of this class initializes the Encoder, Generator, Discriminator and Code Discriminator.
		Arguments:
			image_size	= Height / width of the real images
			n_z		= Dimensionality of the latent space
			n_chan		= Number of channels of the real images
			hiddens		= Number of feature maps in the frist layer of the encoder, generator, discriminator and the number of nodes in the hidden layer of the code discriminator
					  Format:
					  	hiddens	= {'enc': n_enc_hidden,
					  		   'gen': n_gen_hidden,
					  		   'dis': n_dis_hidden,
					  		   'cde': n_cde_hidden
					  		  }
			ngpu		= Number of gpus to be allocated, if to be run on gpu
			loss		= The loss funcion to be used
		"""
		super(ALPHAGAN, self).__init()
		self.Enc_net	= Encoder(image_size, n_z, n_chan, hiddens['enc'], ngpu)
		self.Gen_net	= Generator(image_size, n_z, n_chan, hiddens['gen'], ngpu)
		self.Dis_net	= Discriminator(image_size, n_chan, hiddens['dis'], ngpu)
		self.Cde_dis	= Code_Discriminator(n_z, hiddens['cde'], code_dis_depth, ngpu)
		self.ngpu	= ngpu
		self.n_z	= n_z
		self.image_size	= image_size
		self.n_chan	= n_chan
		self.loss_L1	= nn.L1Loss(size_average=False)
		
	def train(self, dataset, batch_size, n_iters, lmbda, optimizer_details, show_period=50, display_images=True, misc_options=['init_scheme', 'save_model']):
		"""
		Train function of the ALPHAGAN class. This starts training the model.
		Arguments:
			dataset			= Dataset object as from torchvision loader
			batch_size		= batch size to be used throughout the training
			n_iters			= Number of generator iterations to run the training for
			lmbda			= The weighting parameters as depicted in the paper
			optimizer_details	= Dictionary representing the details for optimizers for generator and discriminator
						  Format:
						  	optimizer_details = {'enc':
						  					{'name'		: Name of optimizer,
						  					 'learn_rate'	: learning rate,
						  					 'betas'	: (beta_1, beta_2),	=> Optional, if using Adam/Adamax
						  					 'momentum'	: momentum,		=> Optional, if using momentum SGD/NAG
						  					 'nesterov'	: True/False,		=> Optional, true if using NAG else otherwise
						  					},
						  			     'gen':
						  			     		<SAME AS ABOVE>
						  			     'dis':
						  			     		<SAME AS ABOVE>
						  			     'cde':	
						  			     		<SAME AS ABOVE>
						  			     }
			show_period	(opt)	= Prints the errors with current iteration number every show_period iterations
			display_images	(opt)	= If true, saves the real, reconstructed and generated images from noise every show_period*5 iterations
			misc_options	(opt)	= List of strings.
						  Add 'init_scheme' to the list, if you want to implemente specific initialization schemes
						  Add 'save_model' to the list, if you want to save the model after n_iters iterations of training
		"""
		optimizer_details['enc']['params']	= self.Enc_net.parameters()
		optimizer_details['gen']['params']	= self.Gen_net.parameters()
		optimizer_details['dis']['params']	= self.Dis_net.parameters()
		optimizer_details['cde']['params']	= self.Cde_dis.parameters()
		Enc_optmzr	= u.get_optimizer_with_params(optimizer_details['enc'])
		Gen_optmzr	= u.get_optimizer_with_params(optimizer_details['gen'])
		Dis_optmzr	= u.get_optimizer_with_params(optimizer_details['dis'])
		Cde_optmzr	= u.get_optimizer_with_params(optimizer_details['cde'])
		
		inpt		= t.FloatTensor(batch_size, self.n_chan, self.image_size, self.image_size)
		noise		= t.FloatTensor(batch_size, self.n_z)
		if display_images == True:
			fixed_noise	= t.randn(batch_size, self.n_z)
			
		if 'init_scheme' in misc_options:
			self.Enc_net.apply(u.weight_init_scheme)
			self.Gen_net.apply(u.weight_init_scheme)
			self.Dis_net.apply(u.weight_init_scheme)
			self.Cde_dis.apply(u.weight_init_scheme)

		if self.ngpu > 0:
			inpt	= inpt.cuda()
			noise	= noise.cuda()
			if display_images == True:
				fixed_noise	= fixed_noise.cuda()
				
		d_loader	= d_utils.DataLoader(dataset, batch_size, shuffle=True)
		
		# Train loop
		# Details to be followed:
		# 1. Get all the losses, preferably done by quantified passes over the noise/real data
		# 2. Train the networks as per the method prescribed in the paper. Encoder -> Generator -> Discriminator -> Code Discriminator
		
		gen_iters	= 0
		flag 		= False
		print('Training has started')
		while not flag:
			for i, itr in enumerate(d_loader):
				
				# Reset all grads to zero
				self.Enc_net.zero_grad()
				self.Gen_net.zero_grad()
				self.Dis_net.zero_grad()
				self.Cde_dis.zero_grad()

				# Perform all the passes on the noise and real data
				X, _	= itr
				if inpt.size() != X.size():
					inpt.resize_as_(X)
				inpt.copy_(X)
				inptV	= V(inpt)
				
				# The same number of samples as the input should be taken for noise
				if noise.size(0) != inpt.size():
					noise.resize_(inpt.size(0), noise.size(1), noise.size(2), noise.size(3))
				noise.normal_(0, 1)
				noiseV	= V(noise)
				
				# Get all the passes
				real_encoded	= self.Enc_net(inptV)
				real_reconst	= self.Gen_net(real_encoded)
				fake_generat	= self.Gen_net(noiseV)
				
				# Get the components of the losses
				reconst_loss	= (self.loss_L1(real_reconst, inptV)).mul(lmbda)
				discrim_real	= self.Dis_net(inptV)
				discrim_recons	= self.Dis_net(real_reconst)
				discrim_fake	= self.Dis_net(fake_generat)
				cde_disc_encd	= self.Cde_dis(real_encoded)
				cde_disc_prr	= self.Cde_dis(noiseV)
				
				# Encoder Pass
				encoder_loss	= reconst_loss - t.log(cde_disc_encd).sum()
				encoder_loss.backward(retain_variables=True)
				Enc_optmzr.step()
				
				# Generator Pass
				generator_loss	= reconst_loss - (t.log(discrim_recons) + t.log(discrim_fake)).sum()
				generator_loss.backward(retain_variables=True)
				Gen_optmzr.step()
				
				# Discriminator Pass
				discrimin_loss	= ((t.log(discrim_real) + t.log(1 - discrim_recons) + t.log(1 - discrim_fake)).sum()).mul(-1)
				discrimin_loss.backward(retain_variables=True)
				Dis_optmzr.step()
				
				# Code Discriminator Pass
				cde_disc_loss	= ((t.log(cde_disc_encd) + t.log(1 - cde_disc_prr)).sum()).mul(-1)
				cde_disc_loss.backward(retain_variables=True)
				Cde_optmzr.step()
				
				gen_iters	= gen_iters + 1
				
				# Showing the Progress every show_period iterations
				if gen_iters % show_period == 0:
					print('[{0}/{1}]\tEncoder Loss: \t{2}\tGenerator Loss:\t{3}\tDiscriminator Loss:\t{4}\tCode Discriminator Loss: \t{5}'.format(gen_iters, n_iters, encoder_loss.data[0], generator_loss.data[0], discrimin_loss.data[0], cde_disc_loss.data[0]))
					
				# Saving the real, reconstructed and generated images every show_period*5 iterations
				if display_images == True:
					if gen_iters % (show_period*5) == 0:
						gen_imgs	= self.Gen_net(V(fixed_noise))
						
						# Normalizing the images to look better
						gen_imgs.data	= gen_imgs.data.mul(0.5).add(0.5)
						tv_utils.save_image(gen_imgs.data, 'Generated_images@iteration={0}.png'.format(gen_iters))
						
						X		= X.mul(0.5).add(0.5)
						real_reconst.data	= real_reconst.data.mul(0.5).add(0.5)
						tv_utils.save_image(X, 'Real_images@iteration={0}.png'.format(gen_iters))
						tv_utils.save_image(real_reconst.data, 'Reconstructed_images@iteration={0}.png'.format(gen_iters))
						
				if gen_iters == n_iters:
					flag	= True
					break
					
		if 'save_model' in misc_options and flag == True:
			t.save(self.Enc_net.state_dict(), 'ALPHAGAN_Enc_net_trained_model.pth')
			t.save(self.Gen_net.state_dict(), 'ALPHAGAN_Gen_net_trained_model.pth')
			t.save(self.Dis_net.state_dict(), 'ALPHAGAN_Dis_net_trained_model.pth')
			t.save(self.Cde_dis.state_dict(), 'ALPHAGAN_Cde_dis_trained_model.pth')
			print('Training over and model(s) saved')

		elif flag == True:
			print('Training is over')
				
class Encoder(nn.Module):
	def __init__(self, image_size, n_z, n_chan, n_enc_hiddens, ngpu):
		super(Encoder, self).__init__()

		assert image_size % 16 == 0, "Image size should be a multiple of 16"
		
		self.image_size	= image_size
		self.n_z	= n_z
		self.n_chan	= n_chan
		self.ngpu	= ngpu
		
		self.lin_hid	= n_enc_hiddens['lin']
		n_enc_hidden	= n_enc_hiddens['conv']
		
		# Details to be followed
		# 1. BatchNorm for all layers
		# 2. Except last layer, no other fully connected layers

		layer		= 1
		self.enc_conv	= nn.Sequential()		
		# The first conv layer transforms the image into a set of n_enc_hidden feature maps

		self.enc_conv.add_module('conv_{0}-{1}-{2}'.format(layer, n_chan, n_enc_hidden), nn.Conv2d(n_chan, n_enc_hidden, kernel_size=4, stride=2, padding=1, bias=False))
		self.enc_conv.add_module('batchnorm_{0}-{1}'.format(layer, n_enc_hidden), nn.BatchNorm2d(n_enc_hidden))
		self.enc_conv.add_module('ELU_{0}'.format(layer), nn.ELU(1.67326, inplace=True))
		
		# Keep convolving until the size of the feature map is 4

		cur_size	= image_size//2
		
		while cur_size > 4:
			layer	= layer + 1
			self.enc_conv.add_module('conv_{0}-{1}-{2}'.format(layer, n_enc_hidden, n_enc_hidden*2), nn.Conv2d(n_enc_hidden, n_enc_hidden*2, kernel_size=4, stride=2, padding=1, bias=False))
			self.enc_conv.add_module('batchnorm_{0}-{1}'.format(layer, n_enc_hidden*2), nn.BatchNorm2d(n_enc_hidden*2))
			self.enc_conv.add_module('ELU_{0}'.format(layer), nn.ELU(1.67326, inplace=True))
			cur_size	= cur_size//2
			n_enc_hidden	= n_enc_hidden*2
			
		# Now make the output into a linear vector (per image) representing its encoding of dimensionality n_z

		layer	= layer + 1
		self.enc_conv.add_module('conv_{0}-{1}-{2}'.format(layer, n_enc_hidden, self.lin_hid), nn.Conv2d(n_enc_hidden, self.lin_hid, kernel_size=4, stride=1, padding=0, bias=False))
		self.enc_conv.add_module('batchnorm_{0}-{1}'.format(layer, self.lin_hid), nn.BatchNorm2d(self.lin_hid))
		self.enc_conv.add_module('ELU_{0}'.format(layer), nn.ELU(1.67326, inplace=True))
		
		self.enc_lin	= nn.Sequential()
		self.enc_lin.add_module('linear_1-{0}-{1}'.format(self.lin_hid, n_z), nn.Linear(self.lin_hid, n_z))
		self.enc_lin.add_module('batchnorm_1-{0}'.format(n_z), nn.BatchNorm1d(n_z))
		self.enc_lin.add_module('Tanh_1', nn.Tanh())
		
	def forward(self, input):
		pass_	= input
		if self.ngpu > 0:
			pass_	= nn.parallel.data_parallel(self.enc_conv, pass_, range(0, self.ngpu))
			pass_	= pass_.view(-1, self.lin_hid)
			pass_	= nn.parallel.data_parallel(self.enc_lin, pass_, range(0, self.ngpu))
		else:
			pass_	= self.enc_conv(pass_)
			pass_	= pass_.view(-1, self.lin_hid)
			pass_	= self.enc_lin(pass_)
		
		return pass_
		
class Generator(nn.Module):
	def __init__(self, image_size, n_z, n_chan, n_gen_hiddens, ngpu):
		super(Generator, self).__init__()

		assert image_size % 16 == 0, "Image size should be a multiple of 16"
		
		self.image_size	= image_size
		self.n_z	= n_z
		self.n_chan	= n_chan
		self.ngpu	= ngpu
		
		layer		= 1
		self.gen_lin	= nn.Sequential()
		self.gen_conv	= nn.Sequential()
		
		# Details to be followed
		# 1. BatchNorm for all layers
		# 2. First layer is only linear
		
		self.lin_hid	= n_gen_hiddens['lin']
		self.conv_hid	= n_gen_hiddens['conv']
		
		# The subnet gen_lin performs a fully connected transform over the input 

		self.gen_lin.add_module('linear_{0}-{1}-{2}'.format(layer, n_z, self.lin_hid), nn.Linear(n_z, self.lin_hid))
		self.gen_lin.add_module('batchnorm_{0}-{1}'.format(layer, self.lin_hid), nn.BatchNorm1d(self.lin_hid))
		self.gen_lin.add_module('PReLU_{0}'.format(layer), nn.PReLU())
		
		# The subnet gen_conv converts the output from gen_lin into a tensor of the dimensions of the actual image

		# The first conv layer converts the lin_hid vector into a set of conv_hid feature maps of size 4 each

		layer		= 1
		self.gen_conv.add_module('conv_{0}-{1}-{2}'.format(layer, self.lin_hid, self.conv_hid), nn.ConvTranspose2d(self.lin_hid, self.conv_hid, kernel_size=4, stride=1, padding=0, bias=False))
		self.gen_conv.add_module('batchnorm_{0}-{1}'.format(layer, self.conv_hid), nn.BatchNorm2d(self.conv_hid))
		self.gen_conv.add_module('PReLU_{0}'.format(layer), nn.PReLU())
		
		# Keep enlarging until size of the feature map is image_size//2

		cur_size	= 4
		while cur_size < image_size//2:
			layer	= layer + 1
			self.gen_conv.add_module('conv_{0}-{1}-{2}'.format(layer, self.conv_hid, self.conv_hid//2), nn.ConvTranspose2d(self.conv_hid, self.conv_hid//2, kernel_size=4, stride=2, padding=1, bias=False))
			self.gen_conv.add_module('batchnorm_{0}-{1}'.format(layer, self.conv_hid//2), nn.BatchNorm2d(self.conv_hid//2))
			self.gen_conv.add_module('PReLU_{0}'.format(layer), nn.PReLU())
			cur_size	= cur_size * 2
			self.conv_hid	= self.conv_hid // 2
			
		# The last layer finally produces a tensor of the dimensions of the mini-batch of the actual images

		layer		= layer + 1
		self.gen_conv.add_module('conv_{0}-{1}-{2}'.format(layer, self.conv_hid, n_chan), nn.ConvTranspose2d(self.conv_hid, n_chan, kernel_size=4, stride=2, padding=1, bias=False))
		self.gen_conv.add_module('batchnorm_{0}-{1}'.format(layer, n_chan), nn.BatchNorm2d(n_chan))
		self.gen_conv.add_module('Tanh_{0}'.format(layer), nn.Tanh())
		
	def forward(self, input):
		pass_	= input
		pass_	= pass_.view(-1, self.n_z)
		if self.ngpu > 0:
			pass_	= nn.parallel.data_parallel(self.gen_lin, pass_, range(0, self.ngpu))
			pass_	= pass_.view(-1, self.lin_hid, 1, 1)
			pass_	= nn.parallel.data_parallel(self.gen_conv, pass_, range(0, self.ngpu))
		else:
			pass_	= self.gen_lin(pass_)
			pass_	= pass_.view(-1, self.lin_hid, 1, 1)
			pass_	= self.gen_conv(pass_)
		return pass_
		
class Discriminator(nn.Module):
	def __init__(self, image_size, n_chan, n_dis_hidden, ngpu):
		super(Discriminator, self).__init__()
		
		assert image_size % 16 == 0, "Image size should be a multiple of 16"

		self.image_size	= image_size
		self.n_chan	= n_chan
		self.ngpu	= ngpu

		layer	= 1
		main	= nn.Sequential()
		
		# Details to be followed:
		# 1. Leaky ReLU to be used 
		# 2. BatchNorm for all layers
		
		# This architecture is incredibly similar to DCGAN discriminator architecture

		main.add_module('conv_{0}-{1}-{2}'.format(layer, n_chan, n_dis_hidden), nn.Conv2d(n_chan, n_dis_hidden, kernel_size=4, stride=2, padding=1, bias=False))
		main.add_module('batchnorm_{0}-{1}'.format(layer, n_dis_hidden), nn.BatchNorm2d(n_dis_hidden))
		main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(0.2, inplace=True))
		
		cur_size	= image_size // 2
		while cur_size > 4:
			layer	= layer + 1
			main.add_module('conv_{0}-{1}-{2}'.format(layer, n_dis_hidden, n_dis_hidden*2), nn.Conv2d(n_dis_hidden, n_dis_hidden*2, kernel_size=4, stride=2, padding=1, bias=False))
			main.add_module('batchnorm_{0}-{1}'.format(layer, n_dis_hidden*2), nn.BatchNorm2d(n_dis_hidden*2))
			main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(0.2, inplace=True))
			cur_size	= cur_size//2
			n_dis_hidden	= n_dis_hidden*2
			
		layer	= layer + 1
		main.add_module('conv_{0}-{1}-{2}'.format(layer, n_dis_hidden, 1), nn.Conv2d(n_dis_hidden, 1, kernel_size=4, stride=1, padding=0, bias=False))
		main.add_module('batchnorm_{0}-{1}'.format(layer, 1), nn.BatchNorm2d(1))
		main.add_module('Sigmoid_{0}'.format(layer), nn.Sigmoid())
		
		self.discriminator	= main
		
	def forward(self, input):
		pass_	= input
		if self.ngpu > 0:
			pass_	= nn.parallel.data_parallel(self.discriminator, pass_, range(0, self.ngpu))
		else:
			pass_	= self.discriminator(pass_)
		pass_	= pass_.view(-1, 1)
		return pass_
		
class Code_Discriminator(nn.Module):
	def __init__(self, n_z, n_hidden, depth, ngpu):
		super(Code_Discriminator, self).__init__()
		
		self.n_z	= n_z
		self.ngpu	= ngpu
		main		= nn.Sequential()
		layer		= 1
		
		# Convert the n_z vector represent prior distribution/encoding of image using MLP as instructed in paper

		main.add_module('linear_{0}-{1}-{2}'.format(layer, n_z, n_hidden), nn.Linear(n_z, n_hidden))
		main.add_module('batchnorm_{0}-{1}'.format(layer, n_hidden), nn.BatchNorm1d(n_hidden))
		main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(0.2, inplace=True))
		
		for layer in range(2, depth):
			main.add_module('linear_{0}-{1}-{2}'.format(layer, n_hidden, n_hidden), nn.Linear(n_hidden, n_hidden))
			main.add_module('batchnorm_{0}-{1}'.format(layer, n_hidden), nn.BatchNorm1d(n_hidden))
			main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(0.2, inplace=True))
			
		layer		= layer + 1
		main.add_module('linear_{0}-{1}-{2}'.format(layer, n_hidden, 1), nn.Linear(n_hidden, 1))
		main.add_module('batchnorm_{0}-{1}'.format(layer, 1), nn.BatchNorm1d(1))
		main.add_module('Sigmoid_{0}'.format(layer), nn.Sigmoid())
		
		self.code_dis	= main
		
	def forward(self, input):
		pass_	= input
		if self.ngpu > 0:
			pass_	= nn.parallel.data_parallel(self.code_dis, pass_, range(0, self.ngpu))
		else:
			pass_	= self.code_dis(pass_)
		pass_	= pass_.view(-1, 1)
		return pass_
