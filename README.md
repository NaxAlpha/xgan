# xGAN

xGAN is highly customizable zero-coding GAN implementation for rapid prototyping. You can 
use it quickly train your GAN on your dataset. 

![image](output.gif)

**NOTE**: This repo implements DCGAN and does not include recent breakthroughs in the 
field e.g. Progressive GAN. But I have plan to implement them very soon.

## Getting Started

Follow these steps to train your own GAN:

- Clone the repository `git clone https://github.com/NaxAlpha/xgan.git`
- Install [PyTorch](https://pytorch.org/get-started/locally/) and other requirements 
  `pip install -r requirements.txt`
- Prepare your dataset in `dataset-dir` like this: `dataset-dir/data/abc.png`
- Start training: `python train.py dataset-dir 100 1024 512 256 128 64`
- Duh!

## Options

`train.py` has very diverse set of options available for you to customize. Example usage 
is:

```bash
python train.py <dataset-dir> <network-layers...> [--batch_size=64] [--epochs=100] [--model_dir=None] [--log_iter=10] [--loss_buffer=500] [--n_outputs=3] [--dump_dir=None]
```

Following is parameter documentation:

- `dataset-dir`: Path to images you want to train your GAN on
- `network-layers`: Network architecture from latent space to filters on each layer:
  - Vanilla DCGAN has following parameters: `100 1024 512 256 128 64`
  - First layer value is latent space
  - Size of image is determined number of layers e.g. in case of vanilla we have 6 layers: 2^6 => 64
  - An other example architecture would be for image of size 128x128: `100 1024 512 256 128 64 32`
- `batch_size`: Size of single batch
- `epochs`: Number of epochs for complete iteration of on dataset
- `model_dir`: Path to directory where to save model (skip if you do not want to save model)
- `log_iter`: Number of iterations after which to to save model/output
- `loss_buffer`: Number of values in discriminator/generator loss displayed in output window
- `n_outputs`: Number of images per row displayed in output window
- `dump_dir`: Path where to save output of model (skip not to save output)

### TODO:

- [ ] Sample Jupyter Notebook
- [ ] Settings for Training on Colab
- [ ] Implement Progressive GAN

### Blog Post:

#### [This Icon Does Not Exist — An Application of GANs to Icon Generation](https://medium.com/@NaxAlpha/this-icon-does-not-exist-an-application-of-gans-to-icon-generation-5442f0f867a)
