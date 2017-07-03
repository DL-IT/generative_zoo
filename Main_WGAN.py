import argparse
import torch as t
import numpy as np
import DCGAN as dc
import MLPGAN as mlp
import torch.nn as nn
import torch.utils.data as d_utils
import torchvision.datasets as dset
import torchvision.utils as tv_utils
from torch.autograd import Variable as V
import torchvision.transforms as transforms

t.manual_seed(29)

parser	= argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='mnist | cifar10')
parser.add_argument('--batchsize', type=int, default=100, help='input batch size')
parser.add_argument('--n_chan', type=int, required=True, help='input number of image channels')
parser.add_argument('--n_z', type=int, default=128, help='input dimensionality of latent vector')
parser.add_argument('--img_size', type=int, default=32, help='input the height / width of image')
parser.add_argument('--n_gen_hidden', type=int, default=128, help='input number of feature maps in the first layer of generator')
parser.add_argument('--n_dis_hidden', type=int, default=128, help='input number of feature maps in the first layer of discriminator')
parser.add_argument('--max_gen_iter', type=int, default=200000, help='input the maximum number of generator iterations to run')
parser.add_argument('--gen_lr', type=float, default=5e-05, help='input the learning rate for the generator')
parser.add_argument('--dis_lr', type=float, default=5e-05, help='input the learning rate for the discriminator')
parser.add_argument('--ngpu', type=int, default=0, help='input the number of GPUs required')
parser.add_argument('--dcgan', required=True, type=bool, help='input the requirement of DCGAN')
parser.add_argument('--dis_iter', type=int, default=5, help='input the number of discriminator iterations per generator iteration')
opt	= parser.parse_args()
print(opt)

# Dataset information

if opt.dataset == 'mnist':
	transformations	= [transforms.Scale(opt.img_size), transforms.ToTensor()]
	dataset	= dset.MNIST(root='./{0}_data/'.format(opt.dataset), download=True, transform=transforms.Compose(transformations))
elif opt.dataset == 'cifar10':
	transformations	= [transforms.Scale(opt.img_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
	dataset = dset.CIFAR10(root='./{0}_data/'.format(opt.dataset), download=True, transform=transforms.Compose(transformations))
	
d_loader	= d_utils.DataLoader(dataset, batch_size=opt.batchsize, shuffle=True)

# Hyperparameters

img_size	= opt.img_size
n_z		= opt.n_z
n_G_hidden	= opt.n_gen_hidden
n_D_hidden	= opt.n_dis_hidden
n_chan		= opt.n_chan
b_size		= opt.batchsize
G_lr		= opt.gen_lr
D_lr		= opt.dis_lr
ngpu		= opt.ngpu

# Neural nets

if opt.dcgan == True:
	G_net	= dc.Generator(img_size, n_z, n_chan, n_G_hidden, ngpu)
	D_net	= dc.Discriminator(img_size, n_chan, n_D_hidden, ngpu)
	
	G_net.apply(dc.weight_init_scheme)
	D_net.apply(dc.weight_init_scheme)
else:
	G_net	= mlp.Generator(img_size, n_z, n_chan, n_G_hidden, 128, ngpu)
	D_net	= mlp.Discriminator(img_size, n_chan, n_D_hidden, 128, ngpu)
	
	G_net.apply(mlp.weight_init_scheme)
	D_net.apply(mlp.weight_init_scheme)
	
# Place holders to ensure correct dimensions

inpt	= t.FloatTensor(b_size, n_chan, img_size, img_size)
if opt.dcgan == True:
	noise	= t.FloatTensor(b_size, n_z, 1, 1)
	fixed_noise	= t.randn(b_size, n_z, 1, 1)
else:
	noise	= t.FloatTensor(b_size, n_z)
	fixed_noise	= t.randn(b_size, n_z)
label	= t.FloatTensor(b_size)

# Optimizer and Loss Function

G_optim	= t.optim.RMSprop(G_net.parameters(), lr=G_lr)
D_optim	= t.optim.RMSprop(D_net.parameters(), lr=D_lr)

if ngpu > 0:
	inpt	= inpt.cuda()
	noise	= noise.cuda()
	label	= label.cuda()
	G_net	= G_net.cuda()
	D_net	= D_net.cuda()
	fixed_noise	= fixed_noise.cuda()

gen_iterations	= 0

while True:
	d_iter	= iter(d_loader)
	i	= 0
	while i < len(d_loader):

		# Train discriminator
		j	= 0
		while j < opt.dis_iter and i < len(d_loader):
			j	= j + 1
			data	= d_iter.next()
			i	= i + 1
			
			D_net.zero_grad()

			# Real data
			tr_x, _	= data
			inpt.copy_(tr_x)
			inptv	= V(inpt)
			err_D_r	= D_net(inptv)
			
			# Fake data
			noise.copy_(t.randn(noise.size()))
			noisev	= V(noise)
			fake	= G_net(noisev)
			err_D_f	= D_net(fake)
			
			err_D	= err_D_r - err_D_f
			err_D	= (err_D.mean(0)).view(1)
			err_D.backward()
			D_optim.step()
			
			for params in D_net.parameters():
				params.data.clamp_(-0.01, 0.01)
			
		# Train Generator
		gen_iterations	= gen_iterations + 1
		noise.copy_(t.randn(noise.size()))
		noisev	= V(noise)
		fake	= G_net(noisev)
		err_G	= D_net(fake)
		err_G	= (err_G.mean(0)).view(1)
		err_G.backward()
		G_optim.step()
		
		if gen_iterations == opt.max_gen_iter:
			break
		
		print('[%d/%d] Loss_D: %.4f Loss_G: %.4f' % (gen_iterations, opt.max_gen_iter, err_D.data[0], err_G.data[0]))
			
		if gen_iterations % 500 == 0:
			fake	= G_net(V(fixed_noise))
			tv_utils.save_image(fake.data, 'fake_samples-gen_iter={0}.png'.format(gen_iterations))
