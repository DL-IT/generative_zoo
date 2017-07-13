# generative_zoo

generative_zoo is a repository that provides working implementations of some generative models in [PyTorch](https://pytorch.org). 

#### Available Implementations

| Name of Generative Model | Brief Description | References (if any) |
| ------------------------ | ----------------- | ------------------- |
| Multilayer Perceptron GAN (MLPGAN) | Generative Adversarial Network with MLP Generator Network and Discriminator Network | [Ian Goodfellow et.al.](https://arxiv.org/abs/1406.2661) |
| Deep Convolutional GAN (DCGAN) | Generative Adversarial Network with Deep Convolutional Generator and Discriminator Network | [Alec Radford et.al.](https://arxiv.org/abs/1511.06434) |
| Variational Autoencoder (VAE) | Better performing Autoencoder with a single layer Encoder and Decoder Network | [Kingma et.al.](https://arxiv.org/abs/1312.6114) |
| Wasserstein GAN (WGAN) | Generative Adversarial Network with a different method of training | [Martin Arjovsky et.al.](https://arxiv.org/abs/1701.07875) |
| &#945; GAN | Generative Adversarial Network combined with an Auto-Encoder and a different training strategy | [Mihaela Rosca and Balaji Lakshminarayanan et.al.](https://arxiv.org/abs/1706.04987) |

#### Broken Implementations

+ Unfortunately, the VAEGAN (Combination of a Variational Auto-Encoder and a Generative Adversarial Network) implementation is not training as expected. It will be available soon. Please view the issue [here](https://github.com/DL-IT/generative_zoo/issues/1)

#### Upcoming Implementations

+ The next release of PyTorch will include the usage of the `torch.autograd.grad()` function explicity, allow users to differentiate on their own. The implementation of the WGAN with Gradient Penalty (reference: [Ishaan Gulrajani et.al.](https://arxiv.org/abs/1704.00028)) depends on this release, and hence will be available as quickly as possible after the aforementioned release.


### Requirements

+ **Python 3.x**
+ **PyTorch 0.10 or better**
+ **Torchvision**

### Usage of the implementations

The implementations have been designed to ensure quick and fast usage.

Below is a pipeline:

1. **Import the modules.** 
```py
import xGAN as x
import data_utilities as d_u
```
Please note that `data_utilities` was designed so that the common dataset as extracted from `torchvision` can be used without unnecessary hitches. You can proceed to use your own implementations, but please make sure that the datasets as objects from `torchvision`. Support if required, will be added for non-`torchvision` dataset objects as well.

2. **Get your dataset**
```py
dataset	= y.loader()
```
Currently, `y` is one of _MNIST_, _CIFAR10_ and _LSUN_.
If you dataset is not available on your system, make sure you are connected to a network so that `torchvision` can do the necessary. Parameters to be passed to the loader function depends on the dataset you are using. For any help, please open the Python interpreter in your terminal or your IPython notebook and enter `help(d_u)`.

3. **Initialize all your hyperparameters of the model and the optimizer, create and train your model**
```py
my_gan	= x.xGAN()
my_gan.train()
```
The parameters passed for the object initialization and the train function is specified in detail in the documentation of the functions themselves and vary from model to model. Please open the Python interpreter in your terminal or your IPython notebook and enter `help(my_gan)` or `help(x.xGAN)`.

*That is it, you have your model ready to test*.
