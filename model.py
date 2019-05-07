mport torch
from torch.optim import *
from itertools import chain
from torch.autograd import *
from torch.nn.modules import *
import torchvision.transforms as T
from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


# DC GAN Model
class DCGAN:

    def __init__(self, input_size, latent_size, *layers):
        self.batch_size = None
        self.input_size = input_size
        self.latent_size = latent_size

        self.generator = self.mk_gen(*layers).cuda()
        self.discriminator = self.mk_dis(*reversed(layers)).cuda()

        self.d_opt = Adam(self.discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))
        self.g_opt = Adam(self.generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
        self.loss = BCELoss().cuda()

        self.real_labels = lambda: Variable(torch.ones(self.batch_size, 1)).cuda()
        self.fake_labels = lambda: Variable(torch.zeros(self.batch_size, 1)).cuda()

    def mk_dis(self, *layers):
        model = Sequential(
            Conv2d(1, layers[0], 4, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),
            *chain(*((
                Conv2d(inp, out, 4, 2, 1, bias=False),
                BatchNorm2d(out),
                LeakyReLU(0.2, inplace=True),
            ) for inp, out in zip(layers, layers[1:]))),
            Conv2d(layers[-1], 1, 4, 1, 0, bias=False),
            Sigmoid()
        )
        for param in model.parameters():
            param.data.normal_(0, 0.02)
        return model

    def mk_gen(self, *layers):
        model = Sequential(
            ConvTranspose2d(self.latent_size, layers[0], 4, 1, 0, bias=False),
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
        return Variable(noise).cuda()

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
        real_data = Variable(real_batch).cuda()
        self.batch_size = real_data.size(0)

        e_dis = self.train_discriminator(real_data)
        e_gen = self.train_generator()

        return e_dis.detach().cpu().item(), e_gen.detach().cpu().item()
