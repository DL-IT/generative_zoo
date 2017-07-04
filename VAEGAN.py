import torch as t
import numpy as np
import torch.nn as nn
from torch.autograd import Variable as V

t.manual_seed(29)

class Encoder(nn.Module):
	def __init__(self, image_size, n_chan, n_hidden, n_z, ngpu):
		super(Encoder, self).__init__()
		
		assert image_size % 16 == 0, "Image size should be a multiple of 16"
		
		self.image_size	= image_size
		self.n_chan	= n_chan
		self.n_hidden	= n_hidden
		self.n_z	= n_z
		self.ngpu	= ngpu
		self.encoder	= nn.Sequential()
		
		encoder_layers	= []

		encoder_layers	= make_conv_layer(encoder_layers, n_chan, n_hidden, back_conv=False)
		cur_size = image_size//2
		while cur_size > 8:
			encoder_layers	= make_conv_layer(encoder_layers, n_hidden, n_hidden*2, back_conv=False)
			cur_size	= cur_size//2
			n_hidden	= n_hidden*2
		encoder_layers	= make_conv_layer(encoder_layers, n_hidden, n_hidden*2, back_conv=False, k_s_p=[4,1,0])
			
		for i, layer in enumerate(encoder_layers):
			self.encoder.add_module('component_{0}'.format(i+1), layer)
			
		# determine the size of the last layer's output
		trial	= t.autograd.Variable(t.randn(1, n_chan, image_size, image_size))
		trial	= self.encoder(trial)

		# Fully Connected layer for both mean and covariance separately
		fc_in_enc	= trial.size(1) * trial.size(2) * trial.size(3)
		fc_out_enc	= n_z

		# Fully connected layer for Means
		self.enc_means	= nn.Sequential()
		self.enc_means.add_module('component_1', nn.Linear(fc_in_enc, fc_out_enc))
		self.enc_means.add_module('component_2', nn.BatchNorm1d(fc_out_enc))
		self.enc_means.add_module('component_3', nn.ReLU(True))
		
		# Fully connected layer for Covariances
		self.enc_covs	= nn.Sequential()
		self.enc_covs.add_module('component_1', nn.Linear(fc_in_enc, fc_out_enc))
		self.enc_covs.add_module('component_2', nn.BatchNorm1d(fc_out_enc))
		self.enc_covs.add_module('component_3', nn.ReLU(True))
		
	def forward(self, input):
		pass_	= input
		if self.ngpu > 0:
			pass_	= nn.parallel.data_parallel(self.encoder, pass_, range(0, self.ngpu))
			pass_	= pass_.view(-1, pass_.size(1)*pass_.size(2)*pass_.size(3))
			pass_m	= nn.parallel.data_parallel(self.enc_means, pass_, range(0, self.ngpu))
			pass_c	= nn.parallel.data_parallel(self.enc_covs, pass_, range(0, self.ngpu))
		else:
			pass_	= self.encoder(pass_)
			pass_	= pass_.view(-1, pass_.size(1)*pass_.size(2)*pass_.size(3))
			pass_m	= self.enc_means(pass_)
			pass_c	= self.enc_covs(pass_)
		
		return [pass_m, pass_c]
		
def Parameterizer(input, ngpu):
	pass_		= input
	means	= pass_[0]
	logcovs	= pass_[1]
	
	std	= logcovs.mul(0.5).exp_()
	if ngpu > 0:
		epsilon	= t.cuda.FloatTensor(std.size()).normal_()
	else:
		epsilon	= t.FloatTensor(std.size()).normal_()
		
	epsilon	= V(epsilon)
	epsilon	= epsilon.mul(std).add_(means)
		
	return epsilon.view(epsilon.size(0), -1, 1, 1)
		
class Decoder(nn.Module):
	def __init__(self, image_size, n_chan, n_hidden, n_z, ngpu):
		super(Decoder, self).__init__()
		
		assert image_size % 16 == 0, "Image size should be a multiple of 16"
		
		self.image_size	= image_size
		self.n_chan	= n_chan
		self.n_hidden	= n_hidden
		self.n_z	= n_z
		self.ngpu	= ngpu
		self.decoder	= nn.Sequential()
			
		decoder_layers	= []
		
		decoder_layers	= make_conv_layer(decoder_layers, n_z, n_hidden, back_conv=True, k_s_p=[4,1,0])
		cur_size	= 4
		while cur_size < image_size//2:
			decoder_layers	= make_conv_layer(decoder_layers, n_hidden, n_hidden//2, back_conv=True)
			n_hidden	= n_hidden//2
			cur_size	= cur_size*2
		decoder_layers	= make_conv_layer(decoder_layers, n_hidden, n_chan, back_conv=True, batch_norm=False, activation='Sigmoid')

		for i, layer in enumerate(decoder_layers):
			self.decoder.add_module('component_{0}'.format(i+1), layer)
			
	def forward(self, input):
		pass_	= input
		if self.ngpu > 0:
			pass_	= nn.parallel.data_parallel(self.decoder, pass_, range(0, self.ngpu))
		else:
			pass_	= self.decoder(pass_)
		
		return pass_
		
class Discriminator(nn.Module):
	def __init__(self, image_size, n_chan, n_hidden, ngpu):
		super(Discriminator, self).__init__()
		
		assert image_size % 16 == 0, "Image size should be a multiple of 16"
		
		self.image_size		= image_size
		self.n_chan		= n_chan
		self.n_hidden		= n_hidden
		self.ngpu		= ngpu
		self.discriminator	= nn.Sequential()
		
		discriminator_layers	= []
		discriminator_layers	= make_conv_layer(discriminator_layers, n_chan, n_hidden, back_conv=False, batch_norm=False, activation='LeakyReLU')
		cur_size		= image_size//2
		while cur_size > 4:
			discriminator_layers	= make_conv_layer(discriminator_layers, n_hidden, n_hidden*2, back_conv=False, activation='LeakyReLU')
			cur_size 		= cur_size // 2
			n_hidden		= n_hidden*2
		
		trial	= V(t.randn(1, n_chan, image_size, image_size))
		for layer in discriminator_layers:
			trial	= layer(trial)

		self.fc_in	= trial.size(1) * trial.size(2) * trial.size(3)
		self.fc_out	= 1024
		
		for i, layer in enumerate(discriminator_layers):
			self.discriminator.add_module('component_{0}'.format(i+1), layer)
		
		self.discriminator_lth	= nn.Sequential()
		self.discriminator_lth.add_module('component_1', nn.Linear(self.fc_in, self.fc_out))
		self.discriminator_lth.add_module('component_2', nn.LeakyReLU(0.2, inplace=True))
		
		self.discriminator_last	= nn.Sequential()
		self.discriminator_last.add_module('component_1', nn.Linear(self.fc_out, 1))
		self.discriminator_last.add_module('component_2', nn.Sigmoid())
			
	def forward(self, input):
		pass_	= input
		if self.ngpu > 0:
			pass_	= nn.parallel.data_parallel(self.discriminator, pass_, range(0, self.ngpu))
			pass_	= pass_.view(-1, self.fc_in)
			lth	= nn.parallel.data_parallel(self.discriminator_lth, pass_, range(0, self.ngpu))
			pass_	= nn.parallel.data_parallel(self.discriminator_last, lth, range(0, self.ngpu))
		else:
			pass_	= self.discriminator(pass_)
			pass_	= pass_.view(-1, self.fc_in)
			lth	= self.discriminator_lth(pass_)
			pass_	= self.discriminator_last(lth)
	
		return (pass_.view(-1, 1), lth)
		
class Generator(nn.Module):
	def __init__(self, image_size, n_chan, n_enc_hidden, n_dec_hidden, n_z, ngpu):
		super(Generator, self).__init__()
		
		assert image_size % 16 == 0, "Image size should be a multiple of 16"
		
		self.image_size	= image_size
		self.n_enc_hidd	= n_enc_hidden
		self.n_dec_hidd	= n_dec_hidden
		self.n_z	= n_z
		self.ngpu	= ngpu
		self.encoder	= Encoder(image_size, n_chan, n_enc_hidden, n_z, ngpu)
		self.decoder	= Decoder(image_size, n_chan, n_dec_hidden, n_z, ngpu)
		
	def forward(self, input):
		pass_	= input
		if self.ngpu > 0:
			pass_	= nn.parallel.data_parallel(self.encoder, pass_, range(0, self.ngpu))
			[m, c]	= pass_
			pass_	= Parameterizer(pass_, self.ngpu)
			pass_	= nn.parallel.data_parallel(self.decoder, pass_, range(0, self.ngpu))
		else:
			pass_	= self.encoder(pass_)
			[m, c]	= pass_
			pass_	= Parameterizer(pass_, self.ngpu)
			pass_	= self.decoder(pass_)
		
		return pass_, m, c
		
def make_conv_layer(layer_list, in_dim, out_dim, back_conv, batch_norm=True, activation='ReLU', k_s_p=[4,2,1]):
	k, s, p	= k_s_p[0], k_s_p[1], k_s_p[2]
	if back_conv == False:
		layer_list.append(nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p, bias=False))
	elif back_conv == True:
		layer_list.append(nn.ConvTranspose2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p, bias=False))
		
	if batch_norm == True:
		layer_list.append(nn.BatchNorm2d(out_dim))
	
	if activation == 'ReLU':
		layer_list.append(nn.ReLU(True))
	elif activation == 'Sigmoid':
		layer_list.append(nn.Sigmoid())
	elif activation == 'Tanh':
		layer_list.append(nn.Tanh())
	elif activation == 'LeakyReLU':
		layer_list.append(nn.LeakyReLU(0.2, inplace=True))
	
	return layer_list
	
def weight_init_scheme(mdl):
	name	= mdl.__class__.__name__
	if name.find('Conv') != -1:
		mdl.weight.data.normal_(0.0, 0.02)
	elif name.find('BatchNorm') != -1:
		mdl.weight.data.normal_(1.0, 0.02)
		mdl.bias.data.fill_(0)
	elif name.find('Linear') != -1:
		f_in	= mdl.weight.size(0)
		f_out	= mdl.weight.size(1)
		mdl.weight.data.uniform_(-np.sqrt(6/(f_in + f_out)), np.sqrt(6/(f_in + f_out)))
		mdl.bias.data.fill_(0)
