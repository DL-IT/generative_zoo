import torch as t
import numpy as np
import torch.nn as nn
import torch.utils.data as d_utils
import torchvision.utils as tv_utils
from ..Utilities import utilities as u
from torch.autograd import Variable as V


class VAEGAN(object):
    def __init__(self, image_size, n_z, n_chan, hiddens, ngpu):
        """
        VAEGAN object. This class is a wrapper of a VAEGAN as explained in the paper:
            AUTO-ENCODING BEYOND PIXELS USING A LEARNED SIMILARITY METRIC by Larsen et.al.
        Instance of this class initializes the parameters required for the Encoder, Decoder and Discriminator.
        Arguments:
            image_size = Height / width of the real images
            n_z = Dimensionality of the latent space
            n_chan = Number of channels of the real images
            hiddens = Number of nodes in the hidden layers of the encoder and decoder
                      Format:
                        hiddens = {'enc': n_enc_hidden, 'dec': n_dec_hidden, 'dis': n_dis_hidden}
            ngpu = Number of gpus to be allocated, if to be run on GPU
        """
        super(VAEGAN, self).__init__()

        Gen_net = Generator(image_size, n_chan, hiddens['enc'], hiddens['dec'], n_z, ngpu)
        self.Enc_net = Gen_net.encoder
        self.Dec_net = Gen_net.decoder
        self.Dis_net = Discriminator(image_size, n_chan, hiddens['dis'], ngpu)
        self.ngpu = ngpu
        self.n_z = n_z
        self.image_size = image_size
        self.n_chan = n_chan

    def train(self, dataset, batch_size, n_iters, gamma, optimizer_details, show_period=50,
              display_images=True, misc_options=['init_scheme', 'save_model']):
        """
        Train function of the VAEGAN class. This starts training the model.
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
        optimizer_details['enc']['params'] = self.Enc_net.parameters()
        optimizer_details['dec']['params'] = self.Dec_net.parameters()
        optimizer_details['dis']['params'] = self.Dis_net.parameters()
        E_optmzr = u.get_optimizer_with_params(optimizer_details['enc'])
        Dec_optmzr = u.get_optimizer_with_params(optimizer_details['dec'])
        Dis_optmzr = u.get_optimizer_with_params(optimizer_details['dis'])

        inpt = t.FloatTensor(batch_size, self.n_chan, self.image_size, self.image_size)
        noise = t.FloatTensor(batch_size, self.n_z, 1, 1)
        if display_images:
            fixed_noise = t.randn(batch_size, self.n_z, 1, 1)

        if 'init_scheme' in misc_options:
            self.Enc_net.apply(u.weight_init_scheme)
            self.Dec_net.apply(u.weight_init_scheme)
            self.Dis_net.apply(u.weight_init_scheme)

        if self.ngpu > 0:
            inpt = inpt.cuda()
            noise = noise.cuda()
            if display_images:
                fixed_noise = fixed_noise.cuda()

            self.Enc_net = self.Enc_net.cuda()
            self.Dec_net = self.Dec_net.cuda()
            self.Dis_net = self.Dis_net.cuda()

        d_loader = d_utils.DataLoader(dataset, batch_size, shuffle=True)

        # Train Loop
        # Details to be followed:
        # Minimize Losses w.r.t. every network as specified in paper.
        # Perform simulateneous updates

        gen_iters = 0
        flag = False
        print('Training has started')
        while not flag:
            for i, itr in enumerate(d_loader):

                self.Enc_net.zero_grad()
                self.Dec_net.zero_grad()
                self.Dis_net.zero_grad()

                X, _ = itr
                if inpt.size() != X.size():
                    inpt.resize_(X.size(0), X.size(1), X.size(2), X.size(3))
                inpt.copy_(X)

                inptV = V(inpt)
                if noise.size(0) != inpt.size(0):
                    noise.resize_(inpt.size(0), noise.size(1), noise.size(2), noise.size(3))
                noise.normal_(0, 1)
                noiseV = V(noise)

                # Pass through Encoder to get the means and logcovariances
                # Pass this after reparameterization trick to Decoder
                m_and_logcovs = self.Enc_net(inptV)
                X_recns = self.Dec_net(Parameterizer(m_and_logcovs, self.ngpu))
                X_gen = self.Dec_net(noiseV)

                Dis_real, Dis_real_lth = self.Dis_net(inptV)
                Dis_recns, Dis_recns_lth = self.Dis_net(X_recns)
                Dis_gen, _ = self.Dis_net(X_gen)

                # Assemble the losses
                L_prior = u.KLD(m_and_logcovs)
                L_disll = Recons_loss(Dis_real_lth, Dis_recns_lth)
                L_gan = (t.log(Dis_real) + t.log(1 - Dis_recns) + t.log(1 - Dis_gen)).sum().mul(-1)

                # Assemble the losses per net
                L_enc = L_prior + L_disll
                L_dec = L_disll.mul(gamma) - L_gan
                L_dis = L_gan

                # Perform the backward passes
                L_enc.backward(retain_variables=True)
                L_dec.backward(retain_variables=True)
                L_dis.backward()

                E_optmzr.step()
                Dec_optmzr.step()
                Dis_optmzr.step()

                gen_iters = gen_iters + 1

                # Showing the Progress every show_period iterations
                if gen_iters % show_period == 0:
                    print('[{0}/{1}]\tEncoder Error:\t{2}\tDecoder Error:\t{3}\tDiscriminator Error:\t{4}'
                          .format(gen_iters, n_iters, round(L_enc.data[0], 5),
                                  round(L_dec.data[0], 5), round(L_dis.data[0], 5)))

                # Saving the generated images every show_period*5 iterations
                if display_images:
                    if gen_iters % (show_period*5) == 0:
                        gen_imgs = self.Dec_net(V(fixed_noise))

                        # Normalizing the images to look better
                        if self.n_chan > 1:
                            gen_imgs.data = gen_imgs.data.mul(0.5).add(0.5)
                            X_recns.data = X_recns.data.mul(0.5).add(0.5)
                        tv_utils.save_image(gen_imgs.data,
                                            'VAEGAN_Generated_images@iteration={0}.png'.format(gen_iters))
                        tv_utils.save_image(X_recns.data,
                                            'VAEGAN_Reconstructed_images@iteration={0}.png'.format(gen_iters))

                if gen_iters == n_iters:
                    flag = True
                    break

        if 'save_model' in misc_options and flag:
            t.save(self.Enc_net.state_dict(), 'VAEGAN_Enc_net_trained_model.pth')
            t.save(self.Dec_net.state_dict(), 'VAEGAN_Dec_net_trained_model.pth')
            t.save(self.Dis_net.state_dict(), 'VAEGAN_Dis_net_trained_model.pth')
            print('Training over and model(s) saved')

        elif flag:
            print('Training is over')


class Encoder(nn.Module):
    def __init__(self, image_size, n_chan, n_hidden, n_z, ngpu):
        super(Encoder, self).__init__()

        assert image_size % 16 == 0, "Image size should be a multiple of 16"

        self.image_size = image_size
        self.n_chan = n_chan
        self.n_hidden = n_hidden
        self.n_z = n_z
        self.ngpu = ngpu
        self.encoder = nn.Sequential()

        encoder_layers = []

        encoder_layers = make_conv_layer(encoder_layers, n_chan, n_hidden, back_conv=False)
        cur_size = image_size // 2
        while cur_size > 8:
            encoder_layers = make_conv_layer(encoder_layers, n_hidden, n_hidden * 2, back_conv=False)
            cur_size = cur_size // 2
            n_hidden = n_hidden * 2
        encoder_layers = make_conv_layer(encoder_layers, n_hidden, n_hidden * 2, back_conv=False, k_s_p=[4, 1, 0])

        for i, layer in enumerate(encoder_layers):
            self.encoder.add_module('component_{0}'.format(i + 1), layer)

        # determine the size of the last layer's output
        trial = t.autograd.Variable(t.randn(1, n_chan, image_size, image_size))
        trial = self.encoder(trial)

        # Fully Connected layer for both mean and covariance separately
        fc_in_enc = trial.size(1) * trial.size(2) * trial.size(3)
        fc_out_enc = n_z

        # Fully connected layer for Means
        self.enc_means = nn.Sequential()
        self.enc_means.add_module('component_1', nn.Linear(fc_in_enc, fc_out_enc))
        self.enc_means.add_module('component_2', nn.BatchNorm1d(fc_out_enc))
        self.enc_means.add_module('component_3', nn.ReLU(True))

        # Fully connected layer for Covariances
        self.enc_covs = nn.Sequential()
        self.enc_covs.add_module('component_1', nn.Linear(fc_in_enc, fc_out_enc))
        self.enc_covs.add_module('component_2', nn.BatchNorm1d(fc_out_enc))
        self.enc_covs.add_module('component_3', nn.ReLU(True))

    def forward(self, input):
        pass_ = input
        if self.ngpu > 0:
            pass_ = nn.parallel.data_parallel(self.encoder, pass_, range(0, self.ngpu))
            pass_ = pass_.view(-1, pass_.size(1)*pass_.size(2)*pass_.size(3))
            pass_m = nn.parallel.data_parallel(self.enc_means, pass_, range(0, self.ngpu))
            pass_c = nn.parallel.data_parallel(self.enc_covs, pass_, range(0, self.ngpu))
        else:
            pass_ = self.encoder(pass_)
            pass_ = pass_.view(-1, pass_.size(1)*pass_.size(2)*pass_.size(3))
            pass_m = self.enc_means(pass_)
            pass_c = self.enc_covs(pass_)
        return [pass_m, pass_c]


class Decoder(nn.Module):
    def __init__(self, image_size, n_chan, n_hidden, n_z, ngpu):
        super(Decoder, self).__init__()

        assert image_size % 16 == 0, "Image size should be a multiple of 16"

        self.image_size = image_size
        self.n_chan = n_chan
        self.n_hidden = n_hidden
        self.n_z = n_z
        self.ngpu = ngpu
        self.decoder = nn.Sequential()

        decoder_layers = []

        decoder_layers = make_conv_layer(decoder_layers, n_z, n_hidden, back_conv=True, k_s_p=[4, 1, 0])
        cur_size = 4
        while cur_size < image_size // 2:
            decoder_layers = make_conv_layer(decoder_layers, n_hidden, n_hidden // 2, back_conv=True)
            n_hidden = n_hidden // 2
            cur_size = cur_size * 2
        decoder_layers = make_conv_layer(decoder_layers, n_hidden, n_chan, back_conv=True, batch_norm=False,
                                         activation='Tanh')

        for i, layer in enumerate(decoder_layers):
            self.decoder.add_module('component_{0}'.format(i + 1), layer)

    def forward(self, input):
        pass_ = input
        if self.ngpu > 0:
            pass_ = nn.parallel.data_parallel(self.decoder, pass_, range(0, self.ngpu))
        else:
            pass_ = self.decoder(pass_)
        return pass_


class Discriminator(nn.Module):
    def __init__(self, image_size, n_chan, n_hidden, ngpu):
        super(Discriminator, self).__init__()

        assert image_size % 16 == 0, "Image size should be a multiple of 16"

        self.image_size = image_size
        self.n_chan = n_chan
        self.n_hidden = n_hidden
        self.ngpu = ngpu
        self.discriminator = nn.Sequential()

        discriminator_layers = []
        discriminator_layers = make_conv_layer(discriminator_layers, n_chan, n_hidden, back_conv=False,
                                               batch_norm=False, activation='LeakyReLU')
        cur_size = image_size // 2
        while cur_size > 4:
            discriminator_layers = make_conv_layer(discriminator_layers, n_hidden, n_hidden * 2,
                                                   back_conv=False, activation='LeakyReLU')
            cur_size = cur_size // 2
            n_hidden = n_hidden * 2

        trial = V(t.randn(1, n_chan, image_size, image_size))
        for layer in discriminator_layers:
            trial = layer(trial)

        self.fc_in = trial.size(1) * trial.size(2) * trial.size(3)
        self.fc_out = 512

        for i, layer in enumerate(discriminator_layers):
            self.discriminator.add_module('component_{0}'.format(i + 1), layer)

        self.discriminator_lth = nn.Sequential()
        self.discriminator_lth.add_module('component_1', nn.Linear(self.fc_in, self.fc_out))
        self.discriminator_lth.add_module('component_2', nn.BatchNorm1d(self.fc_out))
        self.discriminator_lth.add_module('component_3', nn.LeakyReLU(0.2, inplace=True))

        self.discriminator_last = nn.Sequential()
        self.discriminator_last.add_module('component_1', nn.Linear(self.fc_out, 1))
        self.discriminator_last.add_module('component_2', nn.Sigmoid())

    def forward(self, input):
        pass_ = input
        if self.ngpu > 0:
            pass_ = nn.parallel.data_parallel(self.discriminator, pass_, range(0, self.ngpu))
            pass_ = pass_.view(-1, self.fc_in)
            lth = nn.parallel.data_parallel(self.discriminator_lth, pass_, range(0, self.ngpu))
            pass_ = nn.parallel.data_parallel(self.discriminator_last, lth, range(0, self.ngpu))
        else:
            pass_ = self.discriminator(pass_)
            pass_ = pass_.view(-1, self.fc_in)
            lth = self.discriminator_lth(pass_)
            pass_ = self.discriminator_last(lth)

        return (pass_.view(-1, 1), lth)


class Generator(nn.Module):
    def __init__(self, image_size, n_chan, n_enc_hidden, n_dec_hidden, n_z, ngpu):
        super(Generator, self).__init__()

        assert image_size % 16 == 0, "Image size should be a multiple of 16"

        self.image_size = image_size
        self.n_enc_hidd = n_enc_hidden
        self.n_dec_hidd = n_dec_hidden
        self.n_z = n_z
        self.ngpu = ngpu
        self.encoder = Encoder(image_size, n_chan, n_enc_hidden, n_z, ngpu)
        self.decoder = Decoder(image_size, n_chan, n_dec_hidden, n_z, ngpu)

    def forward(self, input):
        pass_ = input
        if self.ngpu > 0:
            pass_ = nn.parallel.data_parallel(self.encoder, pass_, range(0, self.ngpu))
            [m, c] = pass_
            pass_ = Parameterizer(pass_, self.ngpu)
            pass_ = nn.parallel.data_parallel(self.decoder, pass_, range(0, self.ngpu))
        else:
            pass_ = self.encoder(pass_)
            [m, c] = pass_
            pass_ = Parameterizer(pass_, self.ngpu)
            pass_ = self.decoder(pass_)
        return pass_, m, c


def make_conv_layer(layer_list, in_dim, out_dim, back_conv, batch_norm=True, activation='ReLU', k_s_p=[4, 2, 1]):
    k, s, p = k_s_p[0], k_s_p[1], k_s_p[2]
    if not back_conv:
        layer_list.append(nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p, bias=False))
    else:
        layer_list.append(nn.ConvTranspose2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p, bias=False))

    if batch_norm:
        layer_list.append(nn.BatchNorm2d(out_dim))

    if activation == 'ReLU':
        layer_list.append(nn.ReLU(True))
    elif activation == 'Sigmoid':
        layer_list.append(nn.Sigmoid())
    elif activation == 'Tanh':
        layer_list.append(nn.Tanh())
    elif activation == 'LeakyReLU':
        layer_list.append(nn.LeakyReLU(0.2, inplace=True))

    return layer_list


def Parameterizer(input, ngpu):
    pass_ = input
    means = pass_[0]
    logcovs = pass_[1]

    std = logcovs.mul(0.5).exp_()
    if ngpu > 0:
        epsilon = t.cuda.FloatTensor(std.size()).normal_()
    else:
        epsilon = t.FloatTensor(std.size()).normal_()

    epsilon = V(epsilon)
    epsilon = epsilon.mul(std).add_(means)
    return epsilon.view(epsilon.size(0), -1, 1, 1)


def Recons_loss(X, means):
    rec_ls = (X - means).clone()
    rec_ls = rec_ls.pow(2)
    rec_ls = rec_ls + np.log(2 * np.pi)
    rec_ls = rec_ls * 0.5
    return rec_ls.sum()
