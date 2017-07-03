import argparse
import VAE as vae
import numpy as np
import torch as t
import torch.nn as nn
import torch.utils.data as d_utils
import torchvision.datasets as dset
import torchvision.utils as tv_utils
from torch.autograd import Variable as V
import torchvision.transforms as transforms

t.manual_seed(29)

parser	= argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, help='mnist')
parser.add_argument('--batchsize', type=int, default=100, help='input batch size')
parser.add_argument('--n_chan', type=int, default=1, help='input the number of channels')
parser.add_argument('--n_z', type=int, default=100, help='input dimensionality of latent vector')
parser.add_argument('--img_size', type=int, default=28, help='input the height / width of image')
parser.add_argument('--n_enc_hidden', type=int, default=250, help='input the number of hidden nodes in the hidden layer of the encoder')
parser.add_argument('--n_dec_hidden', type=int, default=250, help='input the number of hidden nodes in the hidden layer of the decoder')
parser.add_argument('--max_epochs', type=int, default=20, help='input the maximum number of epochs to run')
parser.add_argument('--lr', type=float, default=1e-03, help='input the learning rate of the model')
parser.add_argument('--ngpu', type=int ,default=0, help='input the number of GPUs required')
opt	= parser.parse_args()
print(opt)

# Dataset information

if opt.dataset == 'mnist':
	transformations	= [transforms.ToTensor()]
	dataset 	= dset.MNIST(root='./{0}_data/'.format(opt.dataset), download=True, transform=transforms.Compose(transformations))
	
elif opt.dataset == 'cifar10':
	transformations = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
	dataset 	= dset.CIFAR10(root='./{0}_data/'.format(opt.dataset), download=True, transform=transforms.Compose(transformations))
	
d_loader	= d_utils.DataLoader(dataset, batch_size=opt.batchsize, shuffle=True)

# Hyperparameters

img_size	= opt.img_size
n_z		= opt.n_z
n_enc_hidden	= opt.n_enc_hidden
n_dec_hidden	= opt.n_dec_hidden
n_chan		= opt.n_chan
b_size		= opt.batchsize
lr		= opt.lr
ngpu		= opt.ngpu

# Neural Nets

VAE_net		= vae.var_auto_encoder(n_chan*img_size*img_size, n_enc_hidden, n_dec_hidden, n_z, ngpu)
VAE_net.apply(vae.weight_init_scheme)

# Place holders to ensure correct dimensions

inpt		= t.FloatTensor(b_size, n_chan*img_size*img_size)

# Optimizer and Loss Function

VAE_optim	= t.optim.Adam(VAE_net.parameters(), lr=lr)

recons_loss	= nn.BCELoss(size_average=False)

def loss(inp_img, rec_img, means, logcovs):
	BCE	= recons_loss(rec_img, inp_img)
	
	KLD	= means.pow(2).add_(logcovs.exp()).mul_(-1).add_(logcovs).add_(1)
	KLD	= t.sum(KLD).mul_(-0.5)
	
	return BCE + KLD
	
if ngpu > 0:
	VAE_net	= VAE_net.cuda()
	inpt	= inpt.cuda()
	
for epoch in range(0, opt.max_epochs):
	for i, itr in enumerate(d_loader):
		
		# Training the VAE
		
		VAE_net.zero_grad()
		
		tr_x, _ = itr
		tr_x_	= tr_x.view(-1, img_size*img_size*n_chan)
		
		inpt.copy_(tr_x_)
		inptv	= V(inpt)
		
		recon_img, means, logcovs	= VAE_net(inptv)
		obj_fn	= loss(inptv, recon_img, means, logcovs)
		
		obj_fn.backward()
		
		VAE_optim.step()
		
		if i % 50 == 0:
			print('[{0}/{1}][{2}/{3}]\tObjective Function: {5}'.format(epoch, opt.max_epochs, i, len(d_loader), obj_fn.data[0]))
		
	tv_utils.save_image(tr_x, 'real_samples_{0}.png'.format(epoch))
	recon_img	= recon_img.view(-1, n_chan, img_size, img_size)
	tv_utils.save_image(recon_img.data, 'reconstructed_samples_{0}.png'.format(epoch))
