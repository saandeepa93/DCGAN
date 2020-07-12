import torch
from torch import nn, save, load

from sys import exit as e


class Generator(nn.Module):
  def __init__(self, nz, ngf, nc, dataset_name):
    super(Generator, self).__init__()
    self.dname = dataset_name
    self.hidden0 = nn.Sequential(
      nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias = False),
      nn.BatchNorm2d(ngf*8),
      nn.ReLU(True)
    )

    self.hidden1 = nn.Sequential(
      nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias = False),
      nn.BatchNorm2d(ngf*4),
      nn.ReLU(True)
    )

    self.hidden2 = nn.Sequential(
      nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias = False),
      nn.BatchNorm2d(ngf*2),
      nn.ReLU(True)
    )

    self.hidden3 = nn.Sequential(
      nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias = False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True)
    )

    self.out_mnist = nn.Sequential(
      nn.ConvTranspose2d(ngf*2, nc, 4, 2, 3, bias = False),
      nn.Tanh()
    )

    self.out_others = nn.Sequential(
      nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
      nn.Tanh()
    )


  def forward(self, x):
    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    if self.dname != 'mnist':
      x = self.hidden3(x)
      x = self.out_others(x)
    else:
      x = self.out_mnist(x)
    return x


  def save(self, path):
    """
    This method saves the model at the given path

    Args: path (string): filepath of the model to be saved
    """
    save(self.state_dict(), path)


  def load(self, path):
    """
    This method loads a (saved) model at the given path

    Args: path (string): filepath of the saved model
    """

    self.load_state_dict(load(path))



class Discriminator(nn.Module):
  def __init__(self, nc, ndf, dataset_name):
    super(Discriminator, self).__init__()
    self.dname = dataset_name
    self.hidden0 = nn.Sequential(
      nn.Conv2d(nc, ndf, 4, 2, 1, bias = False),
      nn.LeakyReLU(0.2, inplace = True)
    )

    self.hidden1 = nn.Sequential(
      nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias = False),
      nn.BatchNorm2d(ndf*2),
      nn.LeakyReLU(0.2, inplace = True)
    )


    self.hidden2 = nn.Sequential(
      nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias = False),
      nn.BatchNorm2d(ndf*4),
      nn.LeakyReLU(0.2, inplace = True)
    )

    self.hidden3 = nn.Sequential(
      nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias = False),
      nn.BatchNorm2d(ndf*8),
      nn.LeakyReLU(0.2, inplace = True)
    )

    self.out_mnist = nn.Sequential(
      nn.Conv2d(ndf*4, 1, 3, 1, 0, bias = False),
      nn.Sigmoid()
    )

    self.out_others = nn.Sequential(
      nn.Conv2d(ndf*8, 1, 4, 1, 0, bias = False),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    if self.dname != 'mnist':
      x = self.hidden3(x)
      x = self.out_others(x)
    else:
      x = self.out_mnist(x)
    return x
