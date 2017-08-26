import os
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
		Function to load torcvision dataset object based on just image size
		Args:
			root		= If you dataset is downloaded and ready to use, mention the location of this folder. Else, the dataset will be downloaded to this location
			image_size	= Size of every image
			normalize	= Requirement to normalize the image. Default is true
	"""
	raise NotImplementedError	
