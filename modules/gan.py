import torch
from torch import nn


class Generator(nn.Module):
  def __init__(self, nz, ngf, nc):
    super(Generator, self).__init__()
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


    self.out = nn.Sequential(
      nn.ConvTranspose2d(ngf*2, nc, 4, 2, 3, bias = False),
      nn.Tanh()
    )


  def forward(self, x):
    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    x = self.out(x)
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
  def __init__(self, nc, ndf):
    super(Discriminator, self).__init__()
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


    self.out = nn.Sequential(
      nn.Conv2d(ndf*4, 1, 3, 1, 0, bias = False),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    x = self.out(x)
    return x
