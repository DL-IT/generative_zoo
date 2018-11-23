import torch as t
import torch.nn as nn
import torch.utils.data as d_utils
import torchvision.utils as tv_utils
from torch.autograd import Variable as V
from ..Utilities import utilities as u


class ALPHAGAN(object):
    def __init__(self, image_size, n_z, n_chan, hiddens, code_dis_depth, ngpu):
        """
        ALPHAGAN object. This class is a wrapper of a generalized ALPHA-GAN as explained in the paper:
        VARIATIONAL APPROACHES FOR AUTO-ENCODER GENERATIVE ADVERSARIAL NETWORKS by
        Mihaela Rosca and Balaji Lakshminarayanan et.al.

        Instance of this class initializes the Encoder, Generator, Discriminator and Code Discriminator.
        Arguments:
            image_size = Height / width of the real images
            n_z = Dimensionality of the latent space
            n_chan = Number of channels of the real images
            hiddens = Number of feature maps in the first layer of the encoder, generator,
                      discriminator and the number of nodes in the hidden layer of the code discriminator
                      Format:
                          hiddens = {'enc':
                                        {'lin'    : n_enc_hidden_for_fully_connected,
                                         'conv'   : n_enc_hidden_for_convolutional_first_layer},
                                     'gen':
                                        {'lin'    : n_gen_hidden_for_fully_connected,
                                         'conv'    : n_gen_hidden_for_convolutional_first_layer},
                                     'dis': n_dis_hidden,
                                     'cde': n_cde_hidden}
            code_dis_depth = Number of layers in the Code Discriminator (since it is an MLP)
            ngpu = Number of gpus to be allocated, if to be run on gpu
            loss = The loss funcion to be used
        """
        super(ALPHAGAN, self).__init__()
        self.Enc_net = Encoder(image_size, n_z, n_chan, hiddens['enc'], ngpu)
        self.Gen_net = Generator(image_size, n_z, n_chan, hiddens['gen'], ngpu)
        self.Dis_net = Discriminator(image_size, n_chan, hiddens['dis'], ngpu)
        self.Cde_dis = Code_Discriminator(n_z, hiddens['cde'], code_dis_depth, ngpu)
        self.ngpu = ngpu
        self.n_z = n_z
        self.image_size = image_size
        self.n_chan = n_chan

    def train(self, dataset, batch_size, n_iters, lmbda, optimizer_details, show_period=50,
              display_images=True, misc_options=['init_scheme', 'save_model']):
        """
        Train function of the ALPHAGAN class. This starts training the model.
        Arguments:
            dataset = torch.utils.data.Dataset instance
            batch_size = batch size to be used throughout the training
            n_iters = Number of generator iterations to run the training for
            lmbda = The weighting parameters as depicted in the paper
            optimizer_details = Dictionary representing the details for optimizers for generator and discriminator
                                Format:
                                optimizer_details = {'enc':
                                                        {'name' : Name of optimizer,
                                                         'learn_rate' : learning rate,
                                                         'betas' : (beta_1, beta_2), => Optional, if using Adam/Adamax
                                                         'momentum' : momentum, => Optional, if using momentum SGD/NAG
                                                         'nesterov' : True/False, => Optional, if using NAG},
                                                     'gen':
                                                       <SAME AS ABOVE>
                                                     'dis':
                                                       <SAME AS ABOVE>
                                                     'cde':
                                                       <SAME AS ABOVE>}
            show_period (opt) = Prints the errors with current iteration number every show_period iterations
            display_images (opt) = If true, saves the real, reconstructed and generated images
                                   from noise every show_period*5 iterations
            misc_options (opt) = List of strings.
                                 - Add 'init_scheme' to the list, if you want to implement
                                   specific initialization schemes
                                 - Add 'save_model' to the list, if you want to save the model
                                   after n_iters iterations of training
        """
        optimizer_details['enc']['params'] = self.Enc_net.parameters()
        optimizer_details['gen']['params'] = self.Gen_net.parameters()
        optimizer_details['dis']['params'] = self.Dis_net.parameters()
        optimizer_details['cde']['params'] = self.Cde_dis.parameters()
        Enc_optmzr = u.get_optimizer_with_params(optimizer_details['enc'])
        Gen_optmzr = u.get_optimizer_with_params(optimizer_details['gen'])
        Dis_optmzr = u.get_optimizer_with_params(optimizer_details['dis'])
        Cde_optmzr = u.get_optimizer_with_params(optimizer_details['cde'])

        inpt = t.FloatTensor(batch_size, self.n_chan, self.image_size, self.image_size)
        noise = t.FloatTensor(batch_size, self.n_z)
        if display_images:
            fixed_noise = t.randn(batch_size, self.n_z)

        if 'init_scheme' in misc_options:
            self.Enc_net.apply(u.weight_init_scheme)
            self.Gen_net.apply(u.weight_init_scheme)
            self.Dis_net.apply(u.weight_init_scheme)
            self.Cde_dis.apply(u.weight_init_scheme)

        if self.ngpu > 0:
            inpt = inpt.cuda()
            noise = noise.cuda()
            if display_images:
                fixed_noise = fixed_noise.cuda()

            self.Enc_net = self.Enc_net.cuda()
            self.Gen_net = self.Gen_net.cuda()
            self.Dis_net = self.Dis_net.cuda()
            self.Cde_dis = self.Cde_dis.cuda()

        d_loader = d_utils.DataLoader(dataset, batch_size, shuffle=True)

        # Train loop
        # Details to be followed:
        # 1. Get all the losses, preferably done by quantified passes over the noise/real data
        # 2. Train the networks as per the method prescribed in the paper.
        #    Encoder -> Generator -> Discriminator -> Code Discriminator

        gen_iters = 0
        flag = False
        print('Training has started')
        while not flag:
            for i, itr in enumerate(d_loader):

                # Reset all grads to zero
                self.Enc_net.zero_grad()
                self.Gen_net.zero_grad()
                self.Dis_net.zero_grad()
                self.Cde_dis.zero_grad()

                # Perform all the passes on the noise and real data
                X, _ = itr
                if inpt.size() != X.size():
                    inpt.resize_(X.size(0), X.size(1), X.size(2), X.size(3))
                inpt.copy_(X)
                inptV = V(inpt)

                # The same number of samples as the input should be taken for noise
                if noise.size(0) != inpt.size(0):
                    noise.resize_(inpt.size(0), noise.size(1))
                noise.normal_(0, 1)
                noiseV = V(noise)

                # Get all the passes and all the components of the losses

                # Encoder Pass
                encoder_loss = (self.Gen_net(self.Enc_net(inptV)) - inptV).abs().mul(lmbda).sum() -\
                    t.log(self.Cde_dis(self.Enc_net(inptV))).sum()
                encoder_loss.backward()

                # Generator Pass
                generator_loss = (self.Gen_net(self.Enc_net(inptV)) - inptV).abs().mul(lmbda).sum() - \
                    (t.log(self.Dis_net(self.Gen_net(self.Enc_net(inptV)))) +
                     t.log(self.Dis_net(self.Gen_net(noiseV)))).sum()
                generator_loss.backward()

                # Discriminator Pass
                discrimin_loss = ((t.log(self.Dis_net(inptV)) +
                                   t.log(1 - self.Dis_net(self.Gen_net(self.Enc_net(inptV)))) +
                                   t.log(1 - self.Dis_net(self.Gen_net(noiseV)))).sum()).mul(-1)
                discrimin_loss.backward()

                # Code Discriminator Pass
                cde_disc_loss = ((t.log(self.Cde_dis(self.Enc_net(inptV))) +
                                  t.log(1 - self.Cde_dis(noiseV))).sum()).mul(-1)
                cde_disc_loss.backward()

                # Perform the updates simultaneously
                Enc_optmzr.step()
                Gen_optmzr.step()
                Dis_optmzr.step()
                Cde_optmzr.step()

                gen_iters = gen_iters + 1

                # Showing the Progress every show_period iterations
                if gen_iters % show_period == 0:
                    print('[{0}/{1}]\tEncoder Loss: \t{2}\tGenerator Loss:\t{3}\t\
                           Discriminator Loss:\t{4}\tCode Discriminator Loss: \t{5}'
                          .format(gen_iters, n_iters, round(encoder_loss.data[0], 5),
                                  round(generator_loss.data[0], 5), round(discrimin_loss.data[0], 5),
                                  round(cde_disc_loss.data[0], 5)))

                # Saving the real, reconstructed and generated images every show_period*5 iterations
                if display_images:
                    if gen_iters % (show_period*5) == 0:
                        gen_imgs = self.Gen_net(V(fixed_noise).detach())

                        # Normalizing the images to look better
                        if self.n_chan > 1:
                            gen_imgs.data = gen_imgs.data.mul(0.5).add(0.5)
                        tv_utils.save_image(gen_imgs.data,
                                            'ALPHAGAN_Generated_images@iteration={0}.png'.format(gen_iters))

                        if self.n_chan > 1:
                            X = X.mul(0.5).add(0.5)
                        real_reconst = self.Gen_net(self.Enc_net(inptV.detach()).detach())
                        tv_utils.save_image(real_reconst.data,
                                            'ALPHAGAN_Reconstructed_images@iteration={0}.png'.format(gen_iters))

                if gen_iters == n_iters:
                    flag = True
                    break

        if 'save_model' in misc_options and flag:
            t.save(self.Enc_net.state_dict(), 'ALPHAGAN_Enc_net_trained_model.pth')
            t.save(self.Gen_net.state_dict(), 'ALPHAGAN_Gen_net_trained_model.pth')
            t.save(self.Dis_net.state_dict(), 'ALPHAGAN_Dis_net_trained_model.pth')
            t.save(self.Cde_dis.state_dict(), 'ALPHAGAN_Cde_dis_trained_model.pth')
            print('Training over and model(s) saved')

        elif flag:
            print('Training is over')


class Encoder(nn.Module):
    def __init__(self, image_size, n_z, n_chan, n_enc_hiddens, ngpu):
        super(Encoder, self).__init__()

        assert image_size % 16 == 0, "Image size should be a multiple of 16"

        self.image_size = image_size
        self.n_z = n_z
        self.n_chan = n_chan
        self.ngpu = ngpu

        self.lin_hid = n_enc_hiddens['lin']
        n_enc_hidden = n_enc_hiddens['conv']

        # Details to be followed
        # 1. BatchNorm for all layers
        # 2. Except last layer, no other fully connected layers

        layer = 1
        self.enc_conv = nn.Sequential()
        # The first conv layer transforms the image into a set of n_enc_hidden feature maps

        self.enc_conv.add_module('conv_{0}-{1}-{2}'.format(layer, n_chan, n_enc_hidden),
                                 nn.Conv2d(n_chan, n_enc_hidden, kernel_size=4, stride=2, padding=1, bias=False))
        self.enc_conv.add_module('batchnorm_{0}-{1}'.format(layer, n_enc_hidden), nn.BatchNorm2d(n_enc_hidden))
        self.enc_conv.add_module('ReLU_{0}'.format(layer), nn.ReLU())

        # Keep convolving until the size of the feature map is 4

        cur_size = image_size // 2

        while cur_size > 4:
            layer = layer + 1
            self.enc_conv.add_module('conv_{0}-{1}-{2}'.format(layer, n_enc_hidden, n_enc_hidden * 2),
                                     nn.Conv2d(n_enc_hidden, n_enc_hidden * 2,
                                               kernel_size=4, stride=2, padding=1, bias=False))
            self.enc_conv.add_module('batchnorm_{0}-{1}'.format(layer, n_enc_hidden * 2),
                                     nn.BatchNorm2d(n_enc_hidden*2))
            self.enc_conv.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
            cur_size = cur_size // 2
            n_enc_hidden = n_enc_hidden * 2

        # Now make the output into a linear vector (per image) representing its encoding of dimensionality n_z

        layer = layer + 1
        self.enc_conv.add_module('conv_{0}-{1}-{2}'.format(layer, n_enc_hidden, self.lin_hid),
                                 nn.Conv2d(n_enc_hidden, self.lin_hid,
                                           kernel_size=4, stride=1, padding=0, bias=False))
        self.enc_conv.add_module('batchnorm_{0}-{1}'.format(layer, self.lin_hid), nn.BatchNorm2d(self.lin_hid))
        self.enc_conv.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))

        self.enc_lin = nn.Sequential()
        self.enc_lin.add_module('linear_1-{0}-{1}'.format(self.lin_hid, n_z), nn.Linear(self.lin_hid, n_z))
        self.enc_lin.add_module('batchnorm_1-{0}'.format(n_z), nn.BatchNorm1d(n_z))

    def forward(self, input):
        pass_ = input
        if self.ngpu > 0:
            pass_ = nn.parallel.data_parallel(self.enc_conv, pass_, range(0, self.ngpu))
            pass_ = pass_.view(-1, self.lin_hid)
            pass_ = nn.parallel.data_parallel(self.enc_lin, pass_, range(0, self.ngpu))
        else:
            pass_ = self.enc_conv(pass_)
            pass_ = pass_.view(-1, self.lin_hid)
            pass_ = self.enc_lin(pass_)
        return pass_


class Generator(nn.Module):
    def __init__(self, image_size, n_z, n_chan, n_gen_hiddens, ngpu):
        super(Generator, self).__init__()

        assert image_size % 16 == 0, "Image size should be a multiple of 16"

        self.image_size = image_size
        self.n_z = n_z
        self.n_chan = n_chan
        self.ngpu = ngpu

        layer = 1
        self.gen_lin = nn.Sequential()
        self.gen_conv = nn.Sequential()

        # Details to be followed
        # 1. BatchNorm for all layers
        # 2. First layer is only linear

        self.lin_hid = n_gen_hiddens['lin']
        self.conv_hid = n_gen_hiddens['conv']

        # The subnet gen_lin performs a fully connected transform over the input

        self.gen_lin.add_module('linear_{0}-{1}-{2}'.format(layer, n_z, self.lin_hid), nn.Linear(n_z, self.lin_hid))
        self.gen_lin.add_module('batchnorm_{0}-{1}'.format(layer, self.lin_hid), nn.BatchNorm1d(self.lin_hid))
        self.gen_lin.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))

        # The subnet gen_conv converts the output from gen_lin into a tensor of the dimensions of the actual image

        # The first conv layer converts the lin_hid vector into a set of conv_hid feature maps of size 4 each

        layer = 1
        self.gen_conv.add_module('conv_{0}-{1}-{2}'.format(layer, self.lin_hid, self.conv_hid),
                                 nn.ConvTranspose2d(self.lin_hid, self.conv_hid,
                                                    kernel_size=4, stride=1, padding=0, bias=False))
        self.gen_conv.add_module('batchnorm_{0}-{1}'.format(layer, self.conv_hid), nn.BatchNorm2d(self.conv_hid))
        self.gen_conv.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))

        # Keep enlarging until size of the feature map is image_size//2

        cur_size = 4
        while cur_size < image_size // 2:
            layer = layer + 1
            self.gen_conv.add_module('conv_{0}-{1}-{2}'.format(layer, self.conv_hid, self.conv_hid // 2),
                                     nn.ConvTranspose2d(self.conv_hid, self.conv_hid // 2,
                                                        kernel_size=4, stride=2, padding=1, bias=False))
            self.gen_conv.add_module('batchnorm_{0}-{1}'.format(layer, self.conv_hid // 2),
                                     nn.BatchNorm2d(self.conv_hid // 2))
            self.gen_conv.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
            cur_size = cur_size * 2
            self.conv_hid = self.conv_hid // 2

        # The last layer finally produces a tensor of the dimensions of the mini-batch of the actual images

        layer = layer + 1
        self.gen_conv.add_module('conv_{0}-{1}-{2}'.format(layer, self.conv_hid, n_chan),
                                 nn.ConvTranspose2d(self.conv_hid, n_chan,
                                                    kernel_size=4, stride=2, padding=1, bias=False))
        self.gen_conv.add_module('Tanh_{0}'.format(layer), nn.Tanh())

    def forward(self, input):
        pass_ = input
        pass_ = pass_.view(-1, self.n_z)
        if self.ngpu > 0:
            pass_ = nn.parallel.data_parallel(self.gen_lin, pass_, range(0, self.ngpu))
            pass_ = pass_.view(-1, self.lin_hid, 1, 1)
            pass_ = nn.parallel.data_parallel(self.gen_conv, pass_, range(0, self.ngpu))
        else:
            pass_ = self.gen_lin(pass_)
            pass_ = pass_.view(-1, self.lin_hid, 1, 1)
            pass_ = self.gen_conv(pass_)
        return pass_


class Discriminator(nn.Module):
    def __init__(self, image_size, n_chan, n_dis_hidden, ngpu):
        super(Discriminator, self).__init__()

        assert image_size % 16 == 0, "Image size should be a multiple of 16"

        self.image_size = image_size
        self.n_chan = n_chan
        self.ngpu = ngpu

        layer = 1
        main = nn.Sequential()

        # Details to be followed:
        # 1. Leaky ReLU to be used
        # 2. BatchNorm for all layers

        # This architecture is incredibly similar to DCGAN discriminator architecture

        main.add_module('conv_{0}-{1}-{2}'.format(layer, n_chan, n_dis_hidden),
                        nn.Conv2d(n_chan, n_dis_hidden, kernel_size=4, stride=2, padding=1, bias=False))
        main.add_module('batchnorm_{0}-{1}'.format(layer, n_dis_hidden), nn.BatchNorm2d(n_dis_hidden))
        main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(0.2, inplace=True))

        cur_size = image_size // 2
        while cur_size > 4:
            layer = layer + 1
            main.add_module('conv_{0}-{1}-{2}'.format(layer, n_dis_hidden, n_dis_hidden * 2),
                            nn.Conv2d(n_dis_hidden, n_dis_hidden * 2,
                                      kernel_size=4, stride=2, padding=1, bias=False))
            main.add_module('batchnorm_{0}-{1}'.format(layer, n_dis_hidden * 2), nn.BatchNorm2d(n_dis_hidden * 2))
            main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(0.2, inplace=True))
            cur_size = cur_size // 2
            n_dis_hidden = n_dis_hidden * 2

        layer = layer + 1
        main.add_module('conv_{0}-{1}-{2}'.format(layer, n_dis_hidden, 1),
                        nn.Conv2d(n_dis_hidden, 1, kernel_size=4, stride=1, padding=0, bias=False))
        main.add_module('Sigmoid_{0}'.format(layer), nn.Sigmoid())

        self.discriminator = main

    def forward(self, input):
        pass_ = input
        if self.ngpu > 0:
            pass_ = nn.parallel.data_parallel(self.discriminator, pass_, range(0, self.ngpu))
        else:
            pass_ = self.discriminator(pass_)
        pass_ = pass_.view(-1, 1)
        return pass_


class Code_Discriminator(nn.Module):
    def __init__(self, n_z, n_hidden, depth, ngpu):
        super(Code_Discriminator, self).__init__()

        self.n_z = n_z
        self.ngpu = ngpu
        main = nn.Sequential()
        layer = 1

        # Convert the n_z vector represent prior distribution/encoding of image using MLP as instructed in paper

        main.add_module('linear_{0}-{1}-{2}'.format(layer, n_z, n_hidden), nn.Linear(n_z, n_hidden))
        main.add_module('batchnorm_{0}-{1}'.format(layer, n_hidden), nn.BatchNorm1d(n_hidden))
        main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(0.2, inplace=True))

        for layer in range(2, depth):
            main.add_module('linear_{0}-{1}-{2}'.format(layer, n_hidden, n_hidden), nn.Linear(n_hidden, n_hidden))
            main.add_module('batchnorm_{0}-{1}'.format(layer, n_hidden), nn.BatchNorm1d(n_hidden))
            main.add_module('LeakyReLU_{0}'.format(layer), nn.LeakyReLU(0.2, inplace=True))

        layer = layer + 1
        main.add_module('linear_{0}-{1}-{2}'.format(layer, n_hidden, 1), nn.Linear(n_hidden, 1))
        main.add_module('Sigmoid_{0}'.format(layer), nn.Sigmoid())

        self.code_dis = main

    def forward(self, input):
        pass_ = input
        if self.ngpu > 0:
            pass_ = pass_.view(-1, self.n_z)
            pass_ = nn.parallel.data_parallel(self.code_dis, pass_, range(0, self.ngpu))
        else:
            pass_ = self.code_dis(pass_)
        pass_ = pass_.view(-1, 1)
        return pass_
