import torch
import torch.nn as nn
import numpy as np


class Flatten(nn.Module):
  def forward(self, x):
    x = x.reshape(x.shape[0], -1)
    return x


class globalSubNet(nn.Module):
  def __init__(self, in_img_sz=64, device='cuda'):
    super(globalSubNet, self).__init__()
    self.in_img_sz = in_img_sz
    self.device = device
    self.net_1 = torch.nn.Sequential()
    self.net_2 = torch.nn.Sequential()
    self.net_1.add_module('flatten', Flatten())
    self.net_1.add_module('fc1', torch.nn.Linear(in_img_sz * in_img_sz * 3,
                                                 384))
    self.net_2.add_module('fc2', torch.nn.Linear(384, 192))
    self.net_2.add_module('leakyRelu-fc2', torch.nn.LeakyReLU(inplace=False))
    self.net_2.add_module('fc3', torch.nn.Linear(192, 192))
    self.net_2.add_module('leakyRelu-fc3', torch.nn.LeakyReLU(inplace=False))
    self.net_2.add_module('dropout', torch.nn.Dropout(p=0.5))
    self.net_2.add_module('out', torch.nn.Linear(192, 3 * 9))

  def forward(self, x):
    latent = self.net_1(x)
    m = self.net_2(latent)
    return m, latent
