# This is a sample main file highlighting the usage of MLPGAN module in MLPGAN.py
# Please edit this file based on your requirements
import sys
import MLPGAN as mlp
import data_utilities as d_u

# Dataset
dset	= sys.argv[1]
root	= sys.argv[2]

# Arguments passed to dataset loaders can be modified based on required usage. Please look at the documentation of the loader functions using 
# help(d_u) command.

if dset == 'mnist':
	dataset	= d_u.MNIST_loader(root=root, image_size=32)
	n_chan	= 1
elif dset == 'cifar10':
	dataset	= d_u.CIFAR10_loader(root=root, image_size=32, normalize=True)
	n_chan	= 3
elif dset == 'lsun':
	dataset	= d_u.LSUN_loader(root=root, image_size=32, classes=['bedroom'], normalize=True)
	n_chan	= 3
	
# MLPGAN object initialization
# Parameters below can be modified
# Please check the documentation using help(mlp.MLPGAN.__init__)
image_size	= 32
n_z		= 128
hiddens		= {'gen':	64,
		   'dis':	64
		  }
depths		= {'gen':	4,
		   'dis':	4
		  }
ngpu		= 1
loss		= 'BCE'

Gen_model	= mlp.MLPGAN(image_size=image_size, n_z=n_z, n_chan=n_chan, hiddens=hiddens, depths=depths, ngpu=ngpu, loss=loss)

# MLPGAN training scheme
# Parameters below can be modified based on required usage.
# Please check the documentation using help(mlp.MLPGAN.train) for more details

batch_size	= 100
n_iters		= 1e05
opt_dets	= {'gen':	{'name'		: 'adam',
				 'learn_rate'	: 1e-04,
				 'betas'	: (0.5, 0.99)
				},
		   'dis':	{'name'		: 'sgd',
		   		 'learn_rate'	: 1e-04,
		   		 'momentum'	: 0.9,
		   		 'nesterov'	: True
		   		}
		  }

# Optional arguments
show_period	= 50
display_images	= True
misc_options	= ['init_scheme', 'save_model']

# Call training
Gen_model.train(dataset=dataset, batch_size=batch_size, n_iters=n_iters, optimizer_details=opt_dets, show_period=show_period, display_images=display_images, misc_options=misc_options)

# Voila, your work is done
