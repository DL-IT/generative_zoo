import argparse
import numpy as np
import torch as t
import DCGAN as dc
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
parser.add_argument('--n_z', type=int, default=100, help='input dimensionality of latent vector')
parser.add_argument('--img_size', type=int, default=32, help='input the height / width of image')
parser.add_argument('--n_gen_hidden', type=int, default=128, help='input number of feature maps in the first layer of generator')
parser.add_argument('--n_dis_hidden', type=int, default=128, help='input number of feature maps in the first layer of discriminator')
parser.add_argument('--max_epochs', type=int, default=20, help='input the maximum number of epochs to run')
parser.add_argument('--gen_lr', type=float, default=2e-04, help='input the learning rate for the generator')
parser.add_argument('--dis_lr', type=float, default=2e-04, help='input the learning rate for the discriminator')
parser.add_argument('--ngpu', type=int, default=0, help='input the number of GPUs required')
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

G_net		= dc.Generator(img_size, n_z, n_chan, n_G_hidden, ngpu)
D_net		= dc.Discriminator(img_size, n_chan, n_D_hidden, ngpu)

G_net.apply(dc.weight_init_scheme)
D_net.apply(dc.weight_init_scheme)

# Place holders to ensure correct dimensions

inpt		= t.FloatTensor(b_size, n_chan, img_size, img_size)
noise		= t.FloatTensor(b_size, n_z, 1, 1)
fixed_noise	= t.randn(b_size, n_z, 1, 1)
label		= t.FloatTensor(b_size)

# Optimizer and Loss Function

G_optim		= t.optim.Adam(G_net.parameters(), lr=G_lr, betas=(0.5, 0.999))
D_optim		= t.optim.Adam(D_net.parameters(), lr=D_lr, betas=(0.5, 0.999))
loss_fn		= nn.BCELoss()

if opt.ngpu > 0:
	D_net.cuda()
	G_net.cuda()
	loss_fn.cuda()
	inpt = inpt.cuda()
	label	= label.cuda()
	noise	= noise.cuda()
	fixed_noise	= fixed_noise.cuda()	

gen_iterations	= 0

for epoch in range(0, opt.max_epochs):
	for i, itr in enumerate(d_loader):

		# Training the discriminator
		D_net.zero_grad()
		
		# Training the discriminator with real data
		tr_x, _	= itr
		
		inpt.copy_(tr_x)
		inptv	= V(inpt)
		label.fill_(1)
		labels	= V(label)
		
		otpt	= D_net(inptv)
		err_D_r	= loss_fn(otpt, labels)
		err_D_r.backward()
		
		# Training the discriminator with fake data
		noise.copy_(t.randn(b_size, n_z, 1, 1))
		noisev	= V(noise)
		fake	= G_net(noisev)
		label.fill_(0)
		labels	= V(label)
		
		otpt	= D_net(fake)
		err_D_f	= loss_fn(otpt, labels)
		err_D_f.backward()
		
		err_D	= err_D_r + err_D_f
		D_optim.step()
		
		# Training the generator
		G_net.zero_grad()
		
		# Actual False is True for the generator for obvious reasons
		label.fill_(1)
		labels	= V(label)
		otpt	= D_net(fake)
		err_G	= loss_fn(otpt, labels)
		err_G.backward()
		
		G_optim.step()
		
		gen_iterations	= gen_iterations + 1
		
		if gen_iterations % 10 == 1:
			print('[{0}/{1}][{2}/{3}]\tLoss_D: {4}\tLoss_G: {5}'.format(epoch, opt.max_epochs, i, len(d_loader), err_D.data[0], err_G.data[0]))
		
		if gen_iterations % 200 == 0:
			tv_utils.save_image(tr_x, 'real_samples.png')
			fake	= G_net(V(fixed_noise))
			tv_utils.save_image(fake.data, 'fake_samples-iteration={0}.png'.format(gen_iterations))
