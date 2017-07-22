import torchvision.datasets as dset
import torchvision.transforms as transforms

def MNIST_loader(root, image_size, display_random=False):
	transformations	= [transforms.Scale(image_size), transforms.ToTensor()]
	mnist_data	= dset.MNIST(root=root, download=True, transform=transforms.Compose(transformations))
	return mnist_data
	
def CIFAR10_loader(root, image_size, display_random=False, normalize=True):
	transformations	= [transforms.Scale(image_size), transforms.ToTensor()]
	if normalize == True:
		transformations.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
	cifar10_data	= dset.CIFAR10(root=root, download=True, transform=transforms.Compose(transformations))
	return cifar10_data
	
def LSUN_loader(root, image_size, classes=['bedroom'], display_random=False, normalize=True):
	transformations	= [transforms.Scale(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()]
	if normalize == True:
		transformations.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
	for c in classes:
		c	= c + '_train'
	lsun_data	= dset.LSUN(db_path=root, classes=classes, transform=transforms.Compose(transformations))
	return lsun_data
