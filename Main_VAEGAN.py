import argparse
import torch as t
import numpy as np
import VAEGAN as vg
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
parser.add_argument('--n_enc_hidden', type=int, default=64, help='input the number of feature maps in the first layer of encoder')
parser.add_argument('--n_dec_hidden', type=int, default=256, help='input the number of feature maps in the first layer of decoder')
parser.add_argument('--n_dis_hidden', type=int, default=128, help='input number of feature maps in the first layer of discriminator')
parser.add_argument('--max_epochs', type=int, default=20, help='input the maximum number of epochs to run')
parser.add_argument('--gen_lr', type=float, default=1e-04, help='input the learning rate for the generator')
parser.add_argument('--dis_lr', type=float, default=1e-04, help='input the learning rate for the discriminator')
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
n_chan		= opt.n_chan
n_enc_hidden	= opt.n_enc_hidden
n_dec_hidden	= opt.n_dec_hidden
n_dis_hidden	= opt.n_dis_hidden
b_size	 	= opt.batchsize
G_lr		= opt.gen_lr
D_lr		= opt.dis_lr
ngpu		= opt.ngpu

# Neural_nets

Enc_net		= vg.Generator(img_size, n_chan, n_enc_hidden, n_dec_hidden, n_z, ngpu).encoder
Dec_net		= vg.Generator(img_size, n_chan, n_enc_hidden, n_dec_hidden, n_z, ngpu).decoder
Dis_net		= vg.Discriminator(img_size, n_chan, n_dis_hidden, ngpu)

Enc_net.apply(vg.weight_init_scheme)
Dec_net.apply(vg.weight_init_scheme)
Dis_net.apply(vg.weight_init_scheme)

# Place holders to ensure correct dimensions

inpt		= t.FloatTensor(b_size, n_chan, img_size, img_size)
noise		= t.FloatTensor(b_size, n_z, 1, 1)
fixed_noise	= t.randn(b_size, n_z, 1, 1)
label		= t.FloatTensor(b_size)

# Optimizers and Loss Functions

Enc_optim	= t.optim.Adam(Enc_net.parameters(), lr=G_lr, betas=(0.5, 0.999))
Dec_optim	= t.optim.Adam(Dec_net.parameters(), lr=G_lr, betas=(0.5, 0.999))
Dis_optim	= t.optim.Adam(Dis_net.parameters(), lr=D_lr, betas=(0.5, 0.999))

loss_BCE	= nn.BCELoss(size_average=False)
loss_MSE	= nn.MSELoss(size_average=False)

def loss_KLD(terms):
	means	= terms[0]
	logcovs	= terms[1]
	KLD	= means.pow(2).add_(logcovs.exp()).mul_(-1).add_(logcovs).add_(1)
	KLD	= t.sum(KLD).mul_(-0.5)
	
	return KLD
	
def loss_Dis_Llike(terms):
	x	= terms[0]
	means	= terms[1]
	logcovs	= terms[2]

	pass_	= x
	pass_	= pass_ - means
	pass_	= pass_.pow(2)
	covs	= logcovs.exp()
	pass_	= pass_.div(covs)
	pass_	= pass_ + logcovs
	pass_	= pass_.add_(2*np.pi)
	pass_	= pass_.mul(-0.5)
	pass_	= (pass_.sum()).view(-1, 1)
	return pass_
	
if ngpu > 0:
	Enc_net.cuda()
	Dec_net.cuda()
	Dis_net.cuda()
	loss_BCE.cuda()
	inpt	= inpt.cuda()
	label	= label.cuda()
	noise	= noise.cuda()
	fixed_noise	= fixed_noise.cuda()

gen_iterations	= 0
for epoch in range(0, opt.max_epochs):
	for i, itr in enumerate(d_loader):
		
		Enc_net.zero_grad()
		Dec_net.zero_grad()
		Dis_net.zero_grad()
		
		# Analogies
		# 1. X 		-> X ==> mini-batch from dataset
		# 2. noise 	-> Z_p ==> random noise from multi-variate Gaussian with 0 mean and I covariance
		# 3. X_recns	-> \tilde{X} ==> Reconstructed X 
		# 4. X_p	-> X_p ==> generated images from Z_p
		# 5. L_prior	-> L_{prior} ==> Kullback-Leibler Divergence
		# 6. L_disll	-> L^{Dis_{l}}_{llike} ==> Style term
		# 7. L_gan	-> L_{GAN} ==> GAN loss

		X, _	= itr
		inpt.copy_(X)
		inptv	= V(inpt)
		label.fill_(1)
		labels	= V(label)
		
		noise.copy_(t.randn(b_size, n_z, 1, 1))
		noisev	= V(noise)
		
		outputs	= Enc_net(inptv)
		
		L_prior	= loss_KLD(outputs)
		X_recns	= Dec_net(vg.Parameterizer(outputs, ngpu))
		
		# fix L^{Dis_{l}}_{llike} term
		Dis_real	= Dis_net(inptv)
		Dis_real_recons	= Dis_net(X_recns)
		if ngpu > 0:
			L_disll	= loss_Dis_Llike([Dis_real, Dis_real_recons, V(t.zeros(Dis_real.size()).cuda())]).mul(-1).sum()
		else:
			L_disll	= loss_Dis_Llike([Dis_real, Dis_real_recons, V(t.zeros(Dis_real.size()))]).mul(-1).sum()	
		
		X_p	= Dec_net(noisev)
		
		L_gan	= (t.log(Dis_real) + t.log(Dis_real_recons.mul(-1).add_(1)) + t.log(Dis_net(X_p).mul(-1).add_(1))).sum()
		
		L	= L_gan + L_prior + L_disll
		L.backward()
		
		Enc_optim.step()
		Dec_optim.step()
		Dis_optim.step()
		
		gen_iterations	= gen_iterations + 1
		
		if gen_iterations % 10 == 1:
			print('[{0}/{1}][{2}/{3}]\tL_prior: {4}\tL_disll: {5}\tL_gan: {6}'.format(epoch, opt.max_epochs, i, len(d_loader), round(L_prior.data[0], 5), round(L_disll.data[0], 5), round(L_gan.data[0], 5)))
				
		if gen_iterations % 200 == 0:
			fake	= Dec_net(V(fixed_noise))
			tv_utils.save_image(fake.data, 'fake_samples_from_noise-iteration={0}.png'.format(gen_iterations))
		
	tv_utils.save_image(X, 'real_samples.png')
	recon_imgs	= Dec_net(vg.Parameterizer(Enc_net(inptv), ngpu))
	tv_utils.save_image(recon_imgs.data, 'reconstructed_samples_{0}.png'.format(epoch))
