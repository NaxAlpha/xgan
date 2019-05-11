from itertools import chain

import torch
from torch.autograd import *
from torch.nn.modules import *
from torch.optim import *
from torchsummary import summary


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# DC GAN Model
class DCGAN:

    def __init__(self, latent_size, *layers):
        self.layers = layers
        self.batch_size = None
        self.latent_size = latent_size

        self.generator = self.mk_gen(*layers).to(device)
        self.discriminator = self.mk_dis(*reversed(layers)).to(device)

        self.d_opt = Adam(self.discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))
        self.g_opt = Adam(self.generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
        self.loss = BCELoss().to(device)

        self.real_labels = lambda: Variable(torch.ones(self.batch_size, 1)).to(device)
        self.fake_labels = lambda: Variable(torch.zeros(self.batch_size, 1)).to(device)

    def mk_dis(self, *layers):
        model = Sequential(
            Conv2d(1, layers[0], 3, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),
            *chain(*((
                Conv2d(inp, out, 3, 2, 1, bias=False),
                BatchNorm2d(out),
                LeakyReLU(0.2, inplace=True),
            ) for inp, out in zip(layers, layers[1:]))),
            Conv2d(layers[-1], 1, 2, 1, 0, bias=False),
            Sigmoid()
        )
        for param in model.parameters():
            param.data.normal_(0, 0.02)
        return model
    
    def summary(self, batch_size=1):
        size = 2 ** (len(self.layers) + 1)
        print('=' * 20, 'Generator', '=' * 20, '[{}, {}, 1, 1]'.format(batch_size, self.latent_size))
        summary(self.generator, (self.latent_size, 1, 1), batch_size, device.type)
        print('=' * 20, 'Discriminator', '=' * 16, '[{}, 1, {}, {}]'.format(batch_size, size, size))
        summary(self.discriminator, (1, size, size), batch_size, device.type)
        print('='*51)

    def mk_gen(self, *layers):
        model = Sequential(
            ConvTranspose2d(self.latent_size, layers[0], 2, 1, 0, bias=False),
            BatchNorm2d(layers[0]),
            ReLU(inplace=True),
            *chain(*((
                ConvTranspose2d(inp, out, 4, 2, 1, bias=False),
                BatchNorm2d(out),
                ReLU(inplace=True),
            ) for inp, out in zip(layers, layers[1:]))),
            ConvTranspose2d(layers[-1], 1, 4, 2, 1, bias=False),
            Tanh()
        )
        for param in model.parameters():
            param.data.normal_(0, 0.02)
        return model

    def latent_noise(self, size=None):
        if not size:
            size = self.batch_size
        noise = torch.randn(size, self.latent_size, 1, 1)
        return Variable(noise).to(device)

    def train_discriminator(self, real_data):
        fake_data = self.generator(self.latent_noise()).detach()
        self.d_opt.zero_grad()

        p_real = self.discriminator(real_data)
        e_real = self.loss(p_real, self.real_labels())
        e_real.backward()

        p_fake = self.discriminator(fake_data)
        e_fake = self.loss(p_fake, self.fake_labels())
        e_fake.backward()

        self.d_opt.step()
        return e_fake + e_real

    def train_generator(self):
        fake_data = self.generator(self.latent_noise())
        self.g_opt.zero_grad()

        prd = self.discriminator(fake_data)
        err = self.loss(prd, self.real_labels())
        err.backward()

        self.g_opt.step()
        return err

    def train_step(self, real_batch):
        real_data = Variable(real_batch).to(device)
        self.batch_size = real_data.size(0)

        e_dis = self.train_discriminator(real_data)
        e_gen = self.train_generator()

        return e_dis.detach().cpu().item(), e_gen.detach().cpu().item()

    def save(self, model_file, **extra):
        torch.save(dict(
            discriminator=self.discriminator.state_dict(),
            generator=self.generator.state_dict(),
            **extra
        ), model_file)

    def load(self, model_file):
        state = torch.load(model_file)
        self.discriminator.load_state_dict(state.pop('discriminator'))
        self.generator.load_state_dict(state.pop('generator'))
        return state


if __name__ == '__main__':
    DCGAN(100, 200, 150, 100, 50, 30, 10, 5).summary()

