import torch
from torch.optim import *
from itertools import chain
from torch.autograd import *
from torch.nn.modules import *
import torchvision.transforms as T
from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from model import DCGAN

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    plt.ion()

    trans = T.Compose([
        T.Grayscale(),
        T.Resize(64),
        T.ToTensor(),
        T.Normalize((0.5, ), (0.5, ))
    ])
    # data = MNIST('data/', True, trans)
    data = ImageFolder('data2', trans)

    loader = DataLoader(data, 32, True)

    gen_loss = []
    dsc_loss = []

    mnist = DCGAN(64, 100, 1024, 512, 256, 128)

    fig = plt.figure(figsize=(15, 5))
    for i in range(10000):
        print('Epoch:', i)
        for j, (batch, _) in enumerate(loader):

            dl, gl = mnist.train_step(batch)
            gen_loss.append(gl)
            dsc_loss.append(dl)

            if j % 10 != 0:
                continue

            n_samples = 1000
            n_items = len(gen_loss)

            axs = fig.add_subplot(131)
            axs.cla()
            pts = 100
            # d_x = max(n_items // n_samples, 1)
            x_xs = np.arange(max(n_items-pts, 1), n_items)

            axs.legend(handles=[
                axs.plot(gen_loss[-pts:], label='Generator Loss')[0],
                axs.plot(dsc_loss[-pts:], label='Discriminator Loss')[0]
            ])

            axs = fig.add_subplot(132)
            sample = mnist.latent_noise(9)
            img = mnist.generator(sample).data.cpu()
            grid = make_grid(img, nrow=3, normalize=True).permute(1, 2, 0).numpy()
            axs.imshow(grid)

            axs = fig.add_subplot(133)
            img = batch[:9]
            grid = make_grid(img, nrow=3, normalize=True).permute(1, 2, 0).numpy()
            axs.imshow(grid)

            plt.show()
            plt.pause(0.1)
