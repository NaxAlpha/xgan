import os
from contextlib import suppress

import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from model import DCGAN


def display_loss(fig, dsc_loss, gen_loss):
    axs = fig.add_subplot(131)
    axs.cla()
    axs.legend(handles=[
        axs.plot(gen_loss, label='Generator Loss')[0],
        axs.plot(dsc_loss, label='Discriminator Loss')[0]
    ])


def display_output(fig, network, rows):
    axs = fig.add_subplot(132)
    sample = network.latent_noise(rows**2)
    img = network.generator(sample).data.cpu()
    grid = make_grid(img, nrow=rows, normalize=True).permute(1, 2, 0).numpy()
    axs.imshow(grid)


def display_samples(fig, batch, rows):
    axs = fig.add_subplot(133)
    img = batch[:rows**2]
    grid = make_grid(img, nrow=rows, normalize=True).permute(1, 2, 0).numpy()
    axs.imshow(grid)


def load_model(network, model_dir):
    ep_id, it_id = 0, 0
    gen_loss, dsc_loss = [], []

    if model_dir:
        files = os.listdir(model_dir)
        files.sort(reverse=True)
        for fn in files:
            model_file = os.path.join(model_dir, fn)
            try:
                state = network.load(model_file)
            except Exception:
                continue
            ep_id = state.get('epoch_id', 0)
            it_id = state.get('iter_id', 0)
            gen_loss = state.get('gen_loss', [])
            dsc_loss = state.get('dsc_loss', [])
            break

    return ep_id, it_id, gen_loss, dsc_loss


def training_loop(dataset_path, *layers, batch_size=64, epochs=100, model_dir=None,
                  log_iter=10, loss_buffer=1000, n_outputs=3, dump_dir=None):
    if model_dir:
        with suppress(FileExistsError):
            os.makedirs(model_dir)

    if dump_dir:
        with suppress(FileExistsError):
            os.makedirs(dump_dir)

    plt.ion()

    dataset = ImageFolder(dataset_path, T.Compose([
        T.Grayscale(),
        T.Resize(2 ** len(layers)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ]))

    loader = DataLoader(dataset, batch_size, True)
    network = DCGAN(*layers)
    ep_id, it_id, gen_loss, dsc_loss = load_model(network, model_dir)

    for ep in range(epochs):
        print('Epoch:', ep)

        for i, (batch, _) in enumerate(loader):
            dl, gl = network.train_step(batch)
            gen_loss.append(gl)
            dsc_loss.append(dl)

            if i % log_iter != 0:
                continue

            if model_dir:
                fn = '{:05d}-{:05d}.pt'.format(ep_id+ep, it_id+i)
                model_file = os.path.join(model_dir, fn)
                network.save(model_file,
                             epoch_id=ep_id+ep, iter_id=it_id+i,
                             gen_loss=gen_loss[-loss_buffer:],
                             dsc_loss=dsc_loss[-loss_buffer:])

            fig = plt.figure(0, figsize=(15, 5))
            display_loss(fig, dsc_loss[-loss_buffer:], gen_loss[-loss_buffer:])
            display_output(fig, network, n_outputs)
            display_samples(fig, batch, n_outputs)
            plt.show()

            if dump_dir:
                fn = '{:05d}-{:05d}.png'.format(ep_id+ep, it_id+i)
                image_file = os.path.join(dump_dir, fn)
                fig.savefig(image_file)

            plt.pause(0.1)


if __name__ == '__main__':
    import fire
    fire.Fire(training_loop)

