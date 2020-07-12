from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch import Tensor
import torch
import torch.nn.functional as F
import os
from skimage import io
import glob
from sys import exit as e


import numpy as np


import modules.util as util


class CelebClass(Dataset):
  def __init__(self, root_folder, size):
    super(CelebClass, self).__init__()
    self.root_folder = root_folder
    self.all_files = glob.glob(os.path.join(self.root_folder, '*.jpg'))
    self.transform = transforms.Compose(
      [transforms.ToPILImage(),
      transforms.Resize((size, size)),
      transforms.ToTensor(),
      transforms.Normalize((0.5), (0.5)),
    ])

  def __len__(self):
    return len(self.all_files)

  def __getitem__(self, idx):
    file_name = self.all_files[idx]
    if os.path.splitext(file_name)[-1] == '.jpg':
      img = io.imread(file_name)
      if self.transform:
        img = self.transform(img)
      return img



