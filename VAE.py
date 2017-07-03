import torch as t
import numpy as np
import torch.nn as nn
from torch.autograd import Variable as V

t.manual_seed(29)

def weight_init_scheme(mdl):
	classname	= mdl.__class__.__name__
	
	if classname.find('Linear') != -1:
		f_in	= mdl.weight.size(0)
		f_out	= mdl.weight.size(1)
		mdl.weight.data.uniform_(-np.sqrt(6/(f_in + f_out)), np.sqrt(6/(f_in + f_out)))
		mdl.bias.data.fill_(0)
			
class var_auto_encoder(nn.Module):
	def __init__(self, n_input, n_enc_hidden, n_dec_hidden, n_z, ngpu):
		super(var_auto_encoder, self).__init__()
		self.enc_w_1	= nn.Linear(n_input, n_enc_hidden)
		self.enc_w_m	= nn.Linear(n_enc_hidden, n_z)
		self.enc_w_c	= nn.Linear(n_enc_hidden, n_z)
		self.dec_w_1	= nn.Linear(n_z, n_dec_hidden)
		self.dec_w_2	= nn.Linear(n_dec_hidden, n_input)
		self.enc_act	= nn.Tanh()
		self.dec_act_1	= nn.Tanh()
		self.dec_act_2	= nn.Sigmoid()
		self.ngpu	= ngpu
		
	def encoder(self, input):
		if self.ngpu > 0:
			pass_1	= nn.parallel.data_parallel(self.enc_w_1, input, range(0, self.ngpu))
			pass_2	= nn.parallel.data_parallel(self.enc_act, pass_1, range(0, self.ngpu))
			pass_m	= nn.parallel.data_parallel(self.enc_w_m, pass_2, range(0, self.ngpu))
			pass_c	= nn.parallel.data_parallel(self.enc_w_c, pass_2, range(0, self.ngpu))
			
		else:
			pass_1	= self.enc_w_1(input)
			pass_2	= self.enc_act(pass_1)
			pass_m	= self.enc_w_m(pass_2)
			pass_c	= self.enc_w_c(pass_2)
			
		return pass_m, pass_c			
		
	def parameterization(self, means, logcovs):
		std	= logcovs.mul(0.5).exp_()
		if self.ngpu > 0:
			epsilon	= t.cuda.FloatTensor(std.size()).normal_()
		else:
			epsilon	= t.FloatTensor(std.size()).normal_()
			
		epsilon	= V(epsilon)
		epsilon	= epsilon.mul(std).add_(means)
		
		return epsilon
		
	def decoder(self, input):
		if self.ngpu > 0:
			pass_1	= nn.parallel.data_parallel(self.dec_w_1, input, range(0, self.ngpu))
			pass_2	= nn.parallel.data_parallel(self.dec_act_1, pass_1, range(0, self.ngpu))
			pass_3	= nn.parallel.data_parallel(self.dec_w_2, pass_2, range(0, self.ngpu))
			pass_4	= nn.parallel.data_parallel(self.dec_act_2, pass_3, range(0, self.ngpu))
		else:
			pass_1	= self.dec_w_1(input)
			pass_2	= self.dec_act_1(pass_1)
			pass_3	= self.dec_w_2(pass_2)
			pass_4	= self.dec_act_2(pass_3)

		return pass_4

	def forward(self, input):
		means, logcovs	= self.encoder(input)
		z_		= self.parameterization(means, logcovs)
		recon_img	= self.decoder(z_)
		
		return recon_img, means, logcovs
