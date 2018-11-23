import torch.nn as nn


class Generator(nn.Module):
    """
        Provides an architecture for Generator for any image based dataset.
        This produces images of size image_size x image_size
        Use this module if you are using a very new dataset.
        Motivated by DCGAN by Alec Radford et al.
    """
    def __init__(self, image_size, n_z, n_chan, n_hidden, ngpu):
        """
        Function to construct a Generator instance
        Args:
            image_size : Height / width of the real images
            n_z : Dimensionality of the latent space
            n_chan : Number of channels of the real images
            n_hidden : Number of feature maps in the first layer of the generator
            ngpu : Number of GPUs to be used
        """
        super(Generator, self).__init__()

        assert image_size % 16 == 0, "Image size should be a multiple of 16"
        assert ngpu >= 0, "Number of GPUs has to be non-negative"
        assert n_z > 0, "Dimensionality of the latent space has to be positive"

        layer = 1
        main = nn.Sequential()

        # Details to be followed:
        # 1. ReLU for activation for all but the last layer
        # 2. Batchnorm for all but the last layer
        # 3. No fully connected layers

        # The first conv layer transforms the noise into a set of n_hidden feature maps

        main.add_module('conv_{0}-{1}-{2}'.format(layer, n_z, n_hidden),
                        nn.ConvTranspose2d(n_z, n_hidden, kernel_size=4, stride=1, padding=0, bias=False))
        main.add_module('batchnorm_{0}-{1}'.format(layer, n_hidden), nn.BatchNorm2d(n_hidden))
        main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))

        # Current feature map size is 4
        cur_size = 4

        # Keep enlarging the feature map until before it reaches the size of the image
        while cur_size < image_size // 2:
            layer = layer + 1
            main.add_module('conv_{0}-{1}-{2}'.format(layer, n_hidden, n_hidden // 2),
                            nn.ConvTranspose2d(n_hidden, n_hidden // 2, kernel_size=4, stride=2, padding=1, bias=False))
            main.add_module('batchnorm_{0}-{1}'.format(layer, n_hidden // 2), nn.BatchNorm2d(n_hidden // 2))
            main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))

            n_hidden = n_hidden // 2
            cur_size = cur_size * 2

        # The last conv layer transforms existing feature maps into n_chan feature maps of the size of the image

        layer = layer + 1
        main.add_module('conv_{0}-{1}-{2}'.format(layer, n_hidden, n_chan),
                        nn.ConvTranspose2d(n_hidden, n_chan, kernel_size=4, stride=2, padding=1, bias=False))
        main.add_module('TanH_{0}'.format(layer), nn.Tanh())

        self.main = main
        self.image_size = image_size
        self.n_z = n_z
        self.n_hidden = n_hidden
        self.n_chan = n_chan
        self.ngpu = ngpu

    def forward(self, input):
        if self.ngpu > 0:
            output = nn.parallel.data_parallel(self.main, input, range(0, self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    """
        Provides an architecture for Discriminator for any image based dataset.
        This discriminates images of size image_size x image_size
        Use this module if you are using a very new dataset.
        Motivated by DCGAN by Alec Radford et al.
    """
    def __init__(self, image_size, n_chan, n_hidden, ngpu):
        """
        Function to construct a Generator instance
        Args:
            image_size : Height / width of the real images
            n_chan : Number of channels of the real images
            n_hidden : Number of feature maps in the first layer of the generator
            ngpu : Number of GPUs to be used
        """
        super(Discriminator, self).__init__()

        assert image_size % 16 == 0, "Image size should be a multiple of 16"
        assert ngpu >= 0, "Number of GPUs has to be non-negative"

        layer = 1
        main = nn.Sequential()

        # Details to be followed:
        # 1. Leaky ReLU activation for all but the first and last layer
        # 2. Batchnorm for all but the last layer
        # 3. No fully connected layers

        # The first conv layer transforms the image into a smaller feature map

        main.add_module('conv_{0}-{1}-{2}'.format(layer, n_chan, n_hidden),
                        nn.Conv2d(n_chan, n_hidden, kernel_size=4, stride=2, padding=1, bias=False))
        main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(0.2, inplace=True))

        # Current feature map size is image_size/2

        cur_size = image_size // 2

        # Keep diminishing the size of feature map until before it reaches 4
        while cur_size > 4:
            layer = layer + 1
            main.add_module('conv_{0}-{1}-{2}'.format(layer, n_hidden, n_hidden * 2),
                            nn.Conv2d(n_hidden, n_hidden * 2, kernel_size=4, stride=2, padding=1, bias=False))
            main.add_module('batchnorm_{0}-{1}'.format(layer, n_hidden * 2), nn.BatchNorm2d(n_hidden * 2))
            main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(0.2, inplace=True))

            cur_size = cur_size // 2
            n_hidden = n_hidden * 2

        # The last conv layer transforms the K x 4 x 4 feature map into a K dimensional output vector

        layer = layer + 1
        main.add_module('conv_{0}-{1}-{2}'.format(layer, n_hidden, 1),
                        nn.Conv2d(n_hidden, 1, kernel_size=4, stride=1, padding=0, bias=False))
        main.add_module('Sigmoid_{0}'.format(layer), nn.Sigmoid())

        self.main = main
        self.image_size = image_size
        self.n_hidden = n_hidden
        self.n_chan = n_chan
        self.ngpu = ngpu

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(0, self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1)
