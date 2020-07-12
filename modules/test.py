import torch
import matplotlib.pyplot as plt
import numpy as np
from sys import exit as e

from modules.gan import Generator
import modules.util as util

def plot_2d(generator, noise):
  size_x = 3
  size_y = 3
  fig, axs = plt.subplots(size_x, size_y, figsize=(12, 9))
  m = 0
  for i in np.arange(size_x):
    for j in np.arange(size_y):
      fake_data = generator(noise).view(9, 28, 28)
      axs[i, j].imshow(fake_data[m].detach(), cmap=plt.cm.gray)
      m+=1
  plt.show()

def test_gan(config):
  generator = Generator(config['hypers']['z'])
  generator.load(config['paths']['model'])
  noise = torch.randn(9, config['hypers']['z'])
  plot_2d(generator, noise)
  e()
  img = generator(noise).view(9, 28, 28)
  plt.imshow(img[0].detach())
  plt.show()