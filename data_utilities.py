import os
import torch as t
import numpy as np
from PIL import Image as I
import torch.utils.data as d_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms

def MNIST_loader(root, image_size):
	"""
		Function to load torchvision dataset object based on just image size
		Args:
			root		= If your dataset is downloaded and ready to use, mention the location of this folder. Else, the dataset will be downloaded to this location
			image_size	= Size of every image
	"""
	transformations	= [transforms.Scale(image_size), transforms.ToTensor()]
	mnist_data	= dset.MNIST(root=root, download=True, transform=transforms.Compose(transformations))
	return mnist_data
	
def CIFAR10_loader(root, image_size, normalize=True):
	"""
		Function to load torchvision dataset object based on just image size
		Args:
			root		= If your dataset is downloaded and ready to use, mention the location of this folder. Else, the dataset will be downloaded to this location
			image_size	= Size of every image
			normalize	= Requirement to normalize the image. Default is true
	"""
	transformations	= [transforms.Scale(image_size), transforms.ToTensor()]
	if normalize == True:
		transformations.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
	cifar10_data	= dset.CIFAR10(root=root, download=True, transform=transforms.Compose(transformations))
	return cifar10_data
	
def LSUN_loader(root, image_size, classes=['bedroom'], normalize=True):
	"""
		Function to load torchvision dataset object based on just image size
		Args:
			root		= If your dataset is downloaded and ready to use, mention the location of this folder. Else, the dataset will be downloaded to this location
			image_size	= Size of every image
			classes		= Default class is 'bedroom'. Other available classes are:
						'bridge', 'church_outdoor', 'classroom', 'conference_room', 'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'
			normalize	= Requirement to normalize the image. Default is true
	"""
	transformations	= [transforms.Scale(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()]
	if normalize == True:
		transformations.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
	for c in classes:
		c	= c + '_train'
	lsun_data	= dset.LSUN(db_path=root, classes=classes, transform=transforms.Compose(transformations))
	return lsun_data
	
def CUB200_2010_loader(root, image_size, normalize=True):
	"""
		Function to load torchvision dataset object based on just image size
		Args:
			root		= If you dataset is downloaded and ready to use, mention the location of this folder. Else, the dataset will be downloaded to this location
			image_size	= Size of every image
			normalize	= Requirement to normalize the image. Default is true
	"""
	if os.path.isdir(root):
		# Check if the tarballs are downloaded
		if 'images.tgz' in os.listdir(root) and 'lists.tgz' in os.listdir(root):
			print("Files already downloaded")
		else:
			existing_files = os.listdir(root)
			for f in existing_files:
				os.remove(root + '/' + f)
				
			os.system('wget http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz -P {0} --no-verbose --show-progress'.format(root))
			os.system('wget http://www.vision.caltech.edu/visipedia-data/CUB-200/lists.tgz -P {0} --no-verbose --show-progress'.format(root))

		# Extract the tarballs		
		if 'raw' not in os.listdir(root):
			os.mkdir('{0}/raw/'.format(root))
		os.system('tar -xzf {0}/images.tgz -C {0}/raw/'.format(root))
		os.system('tar -xzf {0}/lists.tgz -C {0}/raw/'.format(root))
		os.remove('{0}/images.tgz'.format(root))
		os.remove('{0}/lists.tgz'.format(root))
	else:
		os.mkdir(root)
		os.system('wget http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz --no-verbose --show-progress')
		os.system('wget http://www.vision.caltech.edu/visipedia-data/CUB-200/lists.tgz --no-verbose --show-progress')
		os.mkdir(root + '/raw')
		os.system('tar -xzf {0}/images.tgz -C {0}/raw/'.format(root))
		os.system('tar -xzf {0}/lists.tgz -C {0}/raw/'.format(root))
		os.remove('{0}/images.tgz'.format(root))
		os.remove('{0}/lists.tgz'.format(root))
	
	# Get all data
	all_data	= []
	all_labels	= []
	train_files	= np.genfromtxt(root + '/raw/lists/train.txt', dtype=str)
	test_files	= np.genfromtxt(root + '/raw/lists/test.txt', dtype=str)
	img_path	= root + '/raw/images/'
	
	for f in train_files:
		img 	= I.open(img_path + f)
		img	= img.resize((image_size, image_size), I.ANTIALIAS)
		npimg	= np.array(img.getdata()).astype(float)
		npimg	= np.reshape(npimg, (img.size[0], img.size[1], 3))
		all_data.append(npimg)
		all_labels.append(int(f[0:3]) - 1)
		img.close()
		
	for f in test_files:
		img	= I.open(img_path + f)
		img	= img.resize((image_size, image_size), I.ANTIALIAS)
		npimg	= np.array(img.getdata()).astype(float)
		npimg	= np.reshape(npimg, (img.size[0], img.size[1], 3))
		all_data.append(npimg)
		all_labels.append(int(f[0:3]) - 1)
		
	all_data	= np.array(all_data)/255
	all_labels	= np.array(all_labels)
	
	all_data_tnsr	= t.from_numpy(all_data).type(t.FloatTensor)
	all_labels_tnsr	= t.from_numpy(all_labels).type(t.FloatTensor)
	if normalize == True:
		nrmlz		= transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		all_data_tnsr	= nrmlz(all_data_tnsr)

	if not os.path.isdir('{0}/processed'.format(root)):
		os.mkdir('{0}/processed'.format(root))

	os.chdir('{0}/processed'.format(root))
	t.save(all_data_tnsr, 'all_inputs.pt')
	t.save(all_labels_tnsr, 'all_outputs.pt')
	
	return d_utils.TensorDataset(all_data_tnsr, all_labels_tnsr)
