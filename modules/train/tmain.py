import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sys import exit as e

from modules.dataset import ColumbiaClass
from modules.train.gan import Generator, Discriminator
import modules.util as util


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('Batch') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


def train(configs):
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0, 0, 0],
                          std=[1, 1, 1]),
  ])
  #dataset
  dataset = ColumbiaClass(configs['paths']['dataset'], transform)
  dataloader = DataLoader(dataset, batch_size = configs['hypers']['batch_size'], shuffle = True)

  #generator
  generator = Generator(0, configs['hypers']['nz'], configs['hypers']['nz'], configs['hypers']['nc'])
  generator.apply(weights_init)

  #discriminator
  discriminator = Discriminator(0, configs['hypers']['nc'], configs['hypers']['ndf'])
  discriminator.apply(weights_init)

  #Loss
  criterion = nn.BCELoss()

  fixed_noise = torch.randn(64, configs['hypers']['nz'], 1, 1)

  real_label = 1
  fake_label = 0

  #Optimizers
  optimizerG = optim.Adam(generator.parameters(), lr = configs['hypers']['lr'], betas = (configs['hypers']['beta'], 0.999))
  optimizerD = optim.Adam(discriminator.parameters(), lr = configs['hypers']['lr'], betas = (configs['hypers']['beta'], 0.999))


  Gloss = []
  Dloss = []
  img_lst = []
  iters = 0
  print("training GAN...")
  for epoch in tqdm(range(configs['hypers']['epochs'])):
    for i, real_data in enumerate(dataloader, 0):
      b_size = real_data.size(0)
      #Train Discriminator
      ## Real data
      optimizerD.zero_grad()
      label = torch.full((b_size,), real_label, dtype = torch.float)
      output = discriminator(real_data).view(-1)
      lossD_real = criterion(output, label)
      lossD_real.backward()
      dx = output.mean().item()

      ##Fake data
      noise = torch.randn(b_size, configs['hypers']['nz'], 1, 1)
      fake_data = generator(noise)
      label.fill_(fake_label)
      output = discriminator(fake_data.detach()).view(-1)
      lossD_fake = criterion(output, label)
      lossD_fake.backward()
      lossD = lossD_real + lossD_fake
      dg_z1 = output.mean().item()
      optimizerD.step()


      #Train Generator
      optimizerG.zero_grad()
      label.fill_(real_label)
      output = discriminator(fake_data).view(-1)
      lossG = criterion(output, label)
      lossG.backward()
      dg_z2 = output.mean().item()
      optimizerG.step()

      if i % 5 == 0:
        print(f"loss_D:{lossD.item()}\n, loss_G:{lossG.item()}\n, D(x):{dx}\n, D(G(z)):{dg_z1}|{dg_z2}\n")
        # print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  # % (epoch, 1, i, len(dataloader),
                  #    lossD.item(), lossG.item(), dx, dg_z1, dg_z2))
      Gloss.append(lossD.item())
      Dloss.append(lossG.item())

      if iters % 10:
        with torch.no_grad():
          fake_data = generator(fixed_noise).detach()
        img_lst.append(fake_data.permute(0, 2, 3, 1))

    e()