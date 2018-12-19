import torch as t
import torch.nn as nn
import torch.utils.data as d_utils
import torchvision.utils as tv_utils
from ..Utilities import utilities as u
from torch.autograd import Variable as V


class VAE(object):
    def __init__(self, image_size, n_z, n_chan, hiddens, ngpu, loss=['KLD', 'BCE']):
        """
        VAE object. This class is a wrapper of a VAE as explained in the paper:
            AUTO-ENCODING VARIATIONAL BAYES by Kingma et.al.

        Instance of this class initializes the parameters required for the Encoder and Decoder.
        Arguments:
            image_size = Height / width of the real images
            n_z = Dimensionality of the latent space
            n_chan = Number of channels of the real images
            hiddens = Number of nodes in the hidden layers of the encoder and decoder
                      Format:
                        hiddens = {'enc': n_enc_hidden,
                                   'dec': n_dec_hidden}
            ngpu = Number of gpus to be allocated, if to be run on gpu
            loss = The loss function to be used. For multiple losses, add them in a list
        """
        super(VAE, self).__init__()
        self.vae_net = vae(image_size, n_z, n_chan, hiddens['enc'], hiddens['dec'], ngpu)
        self.ngpu = ngpu
        self.n_z = n_z
        self.image_size = image_size
        self.n_chan = n_chan
        if 'BCE' in loss:
            self.recons_loss = nn.BCELoss(size_average=False)
        elif 'MSE' in loss:
            self.recons_loss = nn.MSELoss()
        self.KLD_loss = u.KLD

    def train(self, dataset, batch_size, n_iters, optimizer_details, show_period=50,
              display_images=True, misc_options=['init_scheme', 'save_model']):
        """
        Train function of the VAE class. This starts training the model.
        Arguments:
            dataset = torch.utils.data.Dataset instance
            batch_size = batch size to be used throughout the training
            n_iters = Number of generator iterations to run the training for
            optimizer_details = Dictionary representing the details for optimizers for generator and discriminator
                                Format:
                                optimizer_details = {'name' : Name of optimizer,
                                                     'learn_rate' : learning rate,
                                                     'betas' : (beta_1, beta_2), => Optional, if using Adam/Adamax
                                                     'momentum' : momentum, => Optional, if using momentum SGD/NAG
                                                     'nesterov' : True/False, => Optional, if using NAG}
            show_period (opt) = Prints the errors with current iteration number every show_period iterations
            display_images (opt) = If true, saves the real, reconstructed and generated images
                                   from noise every show_period*5 iterations
            misc_options (opt) = List of strings.
                                 - Add 'init_scheme' to the list, if you want to implement
                                   specific initialization schemes
                                 - Add 'save_model' to the list, if you want to save the model
                                   after n_iters iterations of training
        """
        optimizer_details['params'] = self.vae_net.parameters()
        optmzr = u.get_optimizer_with_params(optimizer_details)

        inpt = t.FloatTensor(batch_size, self.n_chan, self.image_size, self.image_size)
        if 'init_scheme' in misc_options:
            self.vae_net.apply(u.weight_init_scheme)

        if self.ngpu > 0:
            inpt = inpt.cuda()

            self.vae_net = self.vae_net.cuda()

        d_loader = d_utils.DataLoader(dataset, batch_size, shuffle=True)

        # Train loop
        # Details to be followed
        # 1. Pass the real images to the vae
        # 2. Calculate the loss, and backpropagate w.r.t. loss

        iters = 0
        flag = False
        while not flag:
            for i, itr in enumerate(d_loader):

                self.vae_net.zero_grad()

                X, _ = itr
                if inpt.size() != X.size():
                    inpt.resize_(X.size(0), X.size(1), X.size(2), X.size(3))
                inpt.copy_(X)

                inptV = V(inpt)
                rec_img, means, logcovs = self.vae_net(inptV)
                obj_fn = self.recons_loss(rec_img, inptV) + self.KLD_loss([means, logcovs])
                obj_fn.backward()
                optmzr.step()

                iters = iters + 1

                # Showing the Progress every show_period iterations
                if iters % show_period == 0:
                    print('[{0}/{1}]\tObjective Function:\t{2}'.format(iters, n_iters, round(obj_fn.data[0], 5)))

                # Saving the reconstructed images every show period * 5 iterations
                if display_images:
                    if iters % (show_period * 5) == 0:
                        # Normalizing the images to look better
                        if self.n_chan > 1:
                            rec_img.data = rec_img.data.mul(0.5).add(0.5)
                        rec_img = rec_img.view(-1, self.n_chan, self.image_size, self.image_size)
                        tv_utils.save_image(rec_img.data,
                                            'VAE_Reconstructed_images@iteration={0}.png'.format(iters))

                if iters == n_iters:
                    flag = True
                    break

            if 'save_model' in misc_options and flag:
                t.save(self.vae_net.state_dict(), 'VAE_net_trained_model.pth')
                print('Training over and model(s) saved')

            elif flag:
                print('Training is over')


class vae(nn.Module):
    def __init__(self, image_size, n_z, n_chan, n_enc_hidden, n_dec_hidden, ngpu):
        super(vae, self).__init__()
        n_input = n_chan * image_size * image_size
        self.enc_w_1 = nn.Linear(n_input, n_enc_hidden)
        self.enc_w_m = nn.Linear(n_enc_hidden, n_z)
        self.enc_w_c = nn.Linear(n_enc_hidden, n_z)
        self.dec_w_1 = nn.Linear(n_z, n_dec_hidden)
        self.dec_w_2 = nn.Linear(n_dec_hidden, n_input)
        self.enc_act = nn.Tanh()
        self.dec_act_1 = nn.Tanh()
        self.dec_act_2 = nn.Sigmoid()
        self.ngpu = ngpu

    def encoder(self, input):
        input = input.view(input.size(0), -1)
        if self.ngpu > 0:
            pass_1 = nn.parallel.data_parallel(self.enc_w_1, input, range(0, self.ngpu))
            pass_2 = nn.parallel.data_parallel(self.enc_act, pass_1, range(0, self.ngpu))
            pass_m = nn.parallel.data_parallel(self.enc_w_m, pass_2, range(0, self.ngpu))
            pass_c = nn.parallel.data_parallel(self.enc_w_c, pass_2, range(0, self.ngpu))
        else:
            pass_1 = self.enc_w_1(input)
            pass_2 = self.enc_act(pass_1)
            pass_m = self.enc_w_m(pass_2)
            pass_c = self.enc_w_c(pass_2)
        return pass_m, pass_c

    def parameterization(self, means, logcovs):
        std = logcovs.mul(0.5).exp_()
        if self.ngpu > 0:
            epsilon = t.cuda.FloatTensor(std.size()).normal_()
        else:
            epsilon = t.FloatTensor(std.size()).normal_()

        epsilon = V(epsilon)
        epsilon = epsilon.mul(std).add_(means)
        return epsilon

    def decoder(self, input):
        if self.ngpu > 0:
            pass_1 = nn.parallel.data_parallel(self.dec_w_1, input, range(0, self.ngpu))
            pass_2 = nn.parallel.data_parallel(self.dec_act_1, pass_1, range(0, self.ngpu))
            pass_3 = nn.parallel.data_parallel(self.dec_w_2, pass_2, range(0, self.ngpu))
            pass_4 = nn.parallel.data_parallel(self.dec_act_2, pass_3, range(0, self.ngpu))
        else:
            pass_1 = self.dec_w_1(input)
            pass_2 = self.dec_act_1(pass_1)
            pass_3 = self.dec_w_2(pass_2)
            pass_4 = self.dec_act_2(pass_3)
        return pass_4

    def forward(self, input):
        means, logcovs = self.encoder(input)
        z_ = self.parameterization(means, logcovs)
        recon_img = self.decoder(z_)
        return recon_img, means, logcovs
