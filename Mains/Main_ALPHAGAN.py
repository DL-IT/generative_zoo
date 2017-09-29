# This is a sample main file highlighting the usage of ALPHAGAN module in ALPHAGAN.py
# Please edit this file based on your requirements
import sys
from generative_zoo.Models import ALPHAGAN as ag
from generative_zoo.Utilities import data_utilities as d_u

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
	
# ALPHAGAN object initialization
# Parameters below can be modified
# Please check the documentation using help(ag.ALPHAGAN.__init__)
image_size	= 32
n_z		= 128
hiddens		= {'enc':	{'lin': 256, 'conv': 128},
		   'gen':	{'lin': 256, 'conv': 128},
		   'dis':	64,
		   'cde':	128
		  }
code_dis_depth	= 3
ngpu		= 1

Gen_model	= ag.ALPHAGAN(image_size=image_size, n_z=n_z, n_chan=n_chan, hiddens=hiddens, code_dis_depth=code_dis_depth, ngpu=ngpu)

# ALPHAGAN training scheme
# Parameters below can be modified based on required usage.
# Please check the documentation using help(ag.ALPHAGAN.train) for more details

batch_size	= 100
n_iters		= 1e05
lmbda		= 0.1
opt_dets	= {'enc':	{'name'		: 'adam',
				 'learn_rate'	: 1e-04,
				 'betas'	: (0.5, 0.99)
				},
		   'gen':	{'name'		: 'sgd',
		   		 'learn_rate'	: 1e-04,
		   		 'momentum'	: 0.9,
		   		 'nesterov'	: True
		   		},
		   'dis':	{'name'		: 'rmsprop',
		   		 'learn_rate'	: 1e-04
		   		}
		   'cde':	{'name'		: 'adam',
		   		 'learn_rate'	: 1e-04,
		   		 'betas'	: (0.5, 0.99)
		   		}
		  }

# Optional arguments
show_period	= 50
display_images	= True
misc_options	= ['init_scheme', 'save_model']

# Call training
Gen_model.train(dataset=dataset, batch_size=batch_size, n_iters=n_iters, lmbda=lmbda, optimizer_details=opt_dets, show_period=show_period, display_images=display_images, misc_options=misc_options)

# Voila, your work is done
