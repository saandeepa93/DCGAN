import torch
from torch.autograd.variable import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import torch.nn.functional as F
import os, shutil
from sys import exit as e

from modules.gan import Discriminator, Generator
from modules.dataset import CelebClass
import modules.util as util
from modules.util import Logger



def mnist_data(out_dir):
  compose = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
  ])
  return datasets.MNIST(root = out_dir, train = True, transform = compose, download = True)


def train_gan(configs, dataset_name):
  if os.path.isdir('./data/'):
    shutil.rmtree('./data/')
  logger = Logger(model_name='DCGAN', data_name='MNIST')
  out_dir = configs['paths']['dataset']

  if dataset_name == 'mnist':
    dataset = mnist_data(out_dir)
  else:
    dataset = CelebClass(out_dir, configs['image']['size'])

  dataloader = DataLoader(dataset, batch_size = configs['hypers']['batch_size'], shuffle = True)
  num_batches = len(dataloader)
  test_noise = torch.randn(16, configs['hypers']['z'], 1, 1)

  discriminator = Discriminator(configs['hypers']['nc'], configs['hypers']['ndf'], dataset_name)
  generator = Generator(configs['hypers']['z'], configs['hypers']['ngf'], configs['hypers']['nc'], dataset_name)

  criterion = nn.BCELoss()

  optimizer_d = optim.Adam(discriminator.parameters(), lr = configs['hypers']['lr'])
  optimizer_g = optim.Adam(generator.parameters(), lr = configs['hypers']['lr'])

  for epoch in range(configs['hypers']['epochs']):
    for b, real_data in enumerate(dataloader, 0):
      if dataset_name == 'mnist':
        real_data, _ = real_data
      b_size = real_data.size(0)

      optimizer_d.zero_grad()
      labels = torch.full((b_size, ), 1, dtype = torch.float)
      output_real = discriminator(real_data).squeeze()
      real_loss = criterion(output_real, labels)
      real_loss.backward()

      labels.fill_(0.0)
      noise = torch.randn(b_size, configs['hypers']['z'], 1, 1)
      fake_data = generator(noise).detach()
      output_fake = discriminator(fake_data).squeeze()
      fake_loss = criterion(output_fake, labels)
      fake_loss.backward()
      optimizer_d.step()
      d_loss = real_loss + fake_loss

      optimizer_g.zero_grad()
      labels.fill_(1.0)
      # labels = torch.full((b_size*2, ), 1, dtype = torch.float)
      noise = torch.randn(b_size, configs['hypers']['z'], 1, 1)
      fake_data = generator(noise)
      output = discriminator(fake_data).squeeze()
      g_loss = criterion(output, labels)
      g_loss.backward()
      optimizer_g.step()
      # Log batch error
      logger.log(d_loss, g_loss, epoch, b, num_batches)        # Display Progress every few batches
      if (b) % 100 == 0:
        test_images = (generator(test_noise)).view(16, configs['hypers']['nc'], configs['image']['size'], configs['image']['size'])
        test_images = test_images.data
        logger.log_images(
            test_images, 16,
            epoch, b, num_batches
        );
        # Display status Logs
        logger.display_status(
            epoch, configs['hypers']['epochs'], b, num_batches,
            d_loss, g_loss, output_real, output_fake
        )
  generator.save(configs['paths']['model'])

