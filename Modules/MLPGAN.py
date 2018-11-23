import torch as t
import torch.nn as nn
import torch.utils.data as d_utils
import torchvision.utils as tv_utils
from ..Utilities import utilities as u
from torch.autograd import Variable as V


class MLPGAN(object):
    def __init__(self, image_size, n_z, n_chan, hiddens, depths, ngpu, loss='BCE'):
        """
        MLPGAN object. This class is a wrapper of a generalized MLPGAN.
        Instance of this class initializes the Generator and the Discriminator.
        Arguments:
            image_size = Height / width of the real images
            n_z = Dimensionality of the latent space
            n_chan = Number of channels of the real images
            hiddens = Number of nodes in the hidden layers of the generator and discriminator
                      Format:
                        hiddens = {'gen': n_gen_hidden,
                                   'dis': n_dis_hidden}
            depths = Number of fully connected layers in the generator and discriminator
                     Format:
                        depths = {'gen': n_gen_depth,
                                  'dis': n_dis_depth}
            ngpu = Number of gpus to allocated, if to be run on gpu
            loss (opt) = The loss function to be used. Default is BCE loss
        """
        super(MLPGAN, self).__init__()
        self.Gen_net = Generator(image_size, n_z, n_chan, hiddens['gen'], depths['gen'], ngpu)
        self.Dis_net = Discriminator(image_size, n_chan, hiddens['dis'], depths['dis'], ngpu)
        self.ngpu = ngpu
        self.n_z = n_z
        self.image_size = image_size
        self.n_chan = n_chan
        if loss == 'BCE':
            self.loss = nn.BCELoss()
        elif loss == 'MSE':
            self.loss = nn.MSELoss()

    def train(self, dataset, batch_size, n_iters, optimizer_details, show_period=50,
              display_images=True, misc_options=['init_scheme', 'save_model']):
        """
        Train function of the MLPGAN class. This starts training the model.
        Arguments:
            dataset = torch.utils.data.Dataset instance
            batch_size = batch size to be used throughout the training
            n_iters = Number of generator iterations to run the training for
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
        optimizer_details['gen']['params'] = self.Gen_net.parameters()
        optimizer_details['dis']['params'] = self.Dis_net.parameters()
        G_optmzr = u.get_optimizer_with_params(optimizer_details['gen'])
        D_optmzr = u.get_optimizer_with_params(optimizer_details['dis'])

        inpt = t.FloatTensor(batch_size, self.n_chan, self.image_size, self.image_size)
        noise = t.FloatTensor(batch_size, self.n_z, 1, 1)
        label = t.FloatTensor(batch_size)
        if display_images:
            fixed_noise = t.randn(batch_size, self.n_z, 1, 1)

        if 'init_scheme' in misc_options:
            self.Gen_net.apply(u.weight_init_scheme)
            self.Dis_net.apply(u.weight_init_scheme)

        if self.ngpu > 0:
            inpt = inpt.cuda()
            noise = noise.cuda()
            label = label.cuda()
            if display_images:
                fixed_noise = fixed_noise.cuda()

            self.Gen_net = self.Gen_net.cuda()
            self.Dis_net = self.Dis_net.cuda()

        d_loader = d_utils.DataLoader(dataset, batch_size, shuffle=True)

        # Train loop
        # Details to be followed:
        # 1. Train the discriminator first. Train the discriminator with reals and then with fakes
        # 2. Train the generator after training the discriminator.

        gen_iters = 0
        flag = False
        print('Training has started')
        while not flag:
            for i, itr in enumerate(d_loader):

                # Training the discriminator
                # We don't want to evaluate the gradients for the Generator during Discriminator training
                self.Dis_net.zero_grad()

                # Training with reals. These are obviously true in the discriminator's POV
                X, _ = itr
                if inpt.size() != X.size():
                    inpt.resize_(X.size(0), X.size(1), X.size(2), X.size(3))
                inpt.copy_(X)
                label.fill_(1)

                inptV = V(inpt)
                labelV = V(label)

                otpt = self.Dis_net(inptV)
                err_D_r = self.loss(otpt, labelV)
                err_D_r.backward()

                # Training with fakes. These are false in the discriminator's POV

                # We want same amount of fake data as real data
                if noise.size(0) != inpt.size(0):
                    noise.resize_(inpt.size(0), noise.size(1), noise.size(2), noise.size(3))
                noise.normal_(0, 1)
                label.fill_(0)

                noiseV = V(noise)
                labelV = V(label)

                X_f = self.Gen_net(noiseV.detach())
                otpt = self.Dis_net(X_f)
                err_D_f = self.loss(otpt, labelV)
                err_D_f.backward()
                err_D = err_D_r + err_D_f
                D_optmzr.step()

                # Training the generator
                # We don't want to evaluate the gradients for the Discriminator during Generator training

                self.Gen_net.zero_grad()

                # The fake are reals in the Generator's POV
                label.fill_(1)

                labelV = V(label)

                X_gen = self.Gen_net(noiseV)
                otpt = self.Dis_net(X_gen)
                err_G = self.loss(otpt, labelV)
                err_G.backward()
                G_optmzr.step()

                gen_iters = gen_iters + 1

                # Showing the Progress every show_period iterations
                if gen_iters % show_period == 0:
                    print('[{0}/{1}]\tDiscriminator Error:\t{2}\tGenerator Error:\t{3}'
                          .format(gen_iters, n_iters, round(err_D.data[0], 5), round(err_G.data[0], 5)))

                # Saving the generated images every show_period*5 iterations
                if display_images:
                    if gen_iters % (show_period * 5) == 0:
                        gen_imgs = self.Gen_net(V(fixed_noise))

                        # Normalizing the images to look better
                        if self.n_chan > 1:
                            gen_imgs.data = gen_imgs.data.mul(0.5).add(0.5)
                        tv_utils.save_image(gen_imgs.data,
                                            'Generated_images@iteration={0}.png'.format(gen_iters))

                if gen_iters == n_iters:
                    flag = True
                    break

        if 'save_model' in misc_options and flag:
            t.save(self.Gen_net.state_dict(), 'MLPGAN_Gen_net_trained_model.pth')
            t.save(self.Dis_net.state_dict(), 'MLPGAN_Dis_net_trained_model.pth')
            print('Training over and model(s) saved')

        elif flag:
            print('Training is over')


class Generator(nn.Module):
    def __init__(self, image_size, n_z, n_chan, n_hidden, depth, ngpu):
        super(Generator, self).__init__()

        self.image_size = image_size
        self.n_z = n_z
        self.n_hidden = n_hidden
        self.n_chan = n_chan
        self.depth = depth
        self.ngpu = ngpu

        layer = 1
        main = nn.Sequential()

        main.add_module('full_connect_{0}_{1}-{2}'.format(layer, n_z, n_hidden), nn.Linear(n_z, n_hidden))
        main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))

        while layer < depth - 1:
            layer = layer + 1
            main.add_module('full_connect_{0}_{1}-{2}'.format(layer, n_hidden, n_hidden),
                            nn.Linear(n_hidden, n_hidden))
            main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))

        layer = layer + 1
        image_dim = image_size * image_size * n_chan
        main.add_module('full_connect_{0}_{1}-{2}'.format(layer, n_hidden, image_dim),
                        nn.Linear(n_hidden, image_dim))
        main.add_module('Tanh_{0}'.format(layer), nn.Tanh())

        self.main = main

    def forward(self, input):
        input = input.view(-1, self.n_z)
        if self.ngpu > 0:
            output = nn.parallel.data_parallel(self.main, input, range(0, self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, self.n_chan, self.image_size, self.image_size)


class Discriminator(nn.Module):
    def __init__(self, image_size, n_chan, n_hidden, depth, ngpu):
        super(Discriminator, self).__init__()

        self.image_size = image_size
        self.n_chan = n_chan
        self.n_hidden = n_hidden
        self.depth = depth
        self.ngpu = ngpu

        layer = 1
        image_dim = image_size * image_size * n_chan
        main = nn.Sequential()

        main.add_module('full_connect_{0}_{1}-{2}'.format(layer, image_dim, n_hidden),
                        nn.Linear(image_dim, n_hidden))
        main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))

        while layer < depth - 1:
            layer = layer + 1
            main.add_module('full_connect_{0}_{1}-{2}'.format(layer, n_hidden, n_hidden),
                            nn.Linear(n_hidden, n_hidden))
            main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))

        layer = layer + 1
        main.add_module('full_connect_{0}_{1}-{2}'.format(layer, n_hidden, 1), nn.Linear(n_hidden, 1))
        main.add_module('Sigmoid_{0}'.format(layer), nn.Sigmoid())

        self.main = main

    def forward(self, input):
        input = input.view(-1, self.n_chan*self.image_size*self.image_size)
        if self.ngpu > 0:
            output = nn.parallel.data_parallel(self.main, input, range(0, self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1)
