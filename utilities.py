import torch as t
import numpy as np

def get_optimizer_with_params(optimizer_details):
	optim_name	= optimizer_details['name']
	learn_rate	= optimizer_details['learn_rate']
	optim_params	= optimizer_details['params']
	
	if 'name' not in optimizer_details.keys() or 'learn_rate' not in optimizer_details.keys() or 'params' not in optimizer_details.keys():
		raise ValueError("Important optimizer details not mentioned")

	if optim_name	== 'adadelta':
		return t.optim.Adadelta(params=optim_params, lr=learn_rate)
	if optim_name	== 'adagrad':
		return t.optim.Adagrad(params=optim_params, lr=learn_rate)
	if optim_name	== 'adam':
		if 'betas' in optimizer_details.keys():
			return t.optim.Adam(params=optim_params, lr=learn_rate, betas=optimizer_details['betas'])
		else:
			return t.optim.Adam(params=optim_params, lr=learn_rate)
	if optim_name	== 'adamax':
		if 'betas' in optimizer_details.keys():
			return t.optim.Adamax(params=optim_params, lr=learn_rate, betas=optimizer_details['betas'])
		else:
			return t.optim.Adamax(params=optim_params, lr=learn_rate)
	if optim_name	== 'rmsprop':
		return t.optim.RMSprop(params=optim_params, lr=learn_rate)
	if optim_name	== 'sgd':
		if 'momentum' in optimizer_details.keys():
			if 'nesterov' in optimizer_details.keys():
				return t.optim.SGD(params=optim_params, lr=learn_rate, momentum=optimizer_details['momentum'], nesterov=optimizer_details['nesterov'])
			else:
				return t.optim.SGD(params=optim_params, lr=learn_rate, momentum=optimizer_details['momentum'])
		else:
			return t.optim.SGD(params=optim_params, lr=learn_rate)
			
def weight_init_scheme(mdl):
	mdl_name	= mdl.__class__.__name__
	if mdl_name.find('Linear') != -1:
		f_in	= mdl.weight.size(0)
		f_out	= mdl.weight.size(1)
		c	= np.sqrt(6/(f_in + f_out))
		mdl.weight.data.uniform_(-c, c)
		mdl.bias.data.fill_(0)
	if mdl_name.find('Conv') != -1:
		mdl.weight.data.normal_(0.0, 0.02)
	if mdl_name.find('BatchNorm') != -1:
		mdl.weight.data.normal_(1.0, 0.02)
		mdl.bias.data.fill_(0)
		
def KLD(means, logcovs):
	l	= means.pow(2)
	l	= l + logcovs.exp()
	l	= l*(-1)
	l 	= l + 1 + logcovs
	l	= (t.sum(l))*(-0.5)
	return l
	
def de_sigmoid(tensor):
	pass_	= tensor
	pass_	= pass_.div(1 - tensor)
	pass_	= t.log(pass_)
	return pass_
