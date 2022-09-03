from .awb_blocks import *
from .global_net import *
import torch.nn as nn

class AWBNet(nn.Module):
  def __init__(self):
    super(AWBNet, self).__init__()
    self.global_net = globalSubNet()
    self.n_channels = 3
    self.encoder_inc = DoubleConvBlock(self.n_channels, 24)
    self.encoder_down1 = DownBlock(24, 24 * 2)
    self.encoder_down1_norm = nn.InstanceNorm2d(24 * 2, affine=True)
    self.encoder_down2 = DownBlock(24 * 2, 24 * 2 * 2)
    self.encoder_down3 = DownBlock(24 * 2 * 2, 24 * 2 * 2 * 2)
    self.encoder_down3_norm = nn.InstanceNorm2d(24 * 2 * 2 * 2, affine=False)
    self.encoder_bridge_down = BridgeDown(24 * 2 * 2 * 2, 24 * 2 * 2 * 2 * 2)
    self.decoder_bridge_up = BridgeUP(24 * 2 * 2 * 2 * 2, 24 * 2 * 2 * 2)
    self.decoder_up1 = UpBlock(24 * 2 * 2 * 2, 24 * 2 * 2)
    self.decoder_up2 = UpBlock(24 * 2 * 2, 24 * 2)
    self.decoder_up3 = UpBlock(24 * 2, 24)
    self.decoder_out = OutputBlock(24, self.n_channels)

  def forward(self, x, histogram):
    m, latent = self.global_net(histogram)
    m = torch.reshape(m, (x.size(0), 9, 3))
    # multiply
    y = x.clone()
    translate = latent.view(x.size(0), -1)[:, :192]
    scale = latent.view(x.size(0), -1)[:, 192:]
    for i in range(m.size(0)):
      temp = torch.mm(self.kernel(torch.reshape(torch.squeeze(
        x[i, :, :, :]), (-1, 3))), torch.squeeze(m[i, :, :]))
      y[i, :, :, :] = torch.reshape(temp, (x.size(1), x.size(2),
                                           x.size(3)))

    x1 = self.encoder_inc(y)
    x2 = self.encoder_down1(x1)
    x3 = self.encoder_down1_norm(x2)
    x3 = self.encoder_down2(x3)
    x4 = self.encoder_down3(x3)
    x5 = self.encoder_down3_norm(x4)
    x5 = x5 * scale.view(x5.shape[0], -1, 1, 1) + translate.view(
      x5.shape[0], -1, 1, 1)
    x5 = self.encoder_bridge_down(x5)
    x6 = self.decoder_bridge_up(x5)
    x6 = self.decoder_up1(x6, x4)
    x6 = self.decoder_up2(x6, x3)
    x6 = self.decoder_up3(x6, x2)
    out = self.decoder_out(x6, x1) + y
    return out, y

  @staticmethod
  def kernel(x):
      return torch.cat((x, x * x, torch.unsqueeze(x[:, 0] * x[:, 1], dim=-1),
                        torch.unsqueeze(x[:, 0] * x[:, 2], dim=-1),
                        torch.unsqueeze(x[:, 1] * x[:, 2], dim=-1)), dim=1)



class AWBNet_wo_R1(nn.Module):
  def __init__(self):
    super(AWBNet_wo_R1, self).__init__()
    self.n_channels = 3
    self.encoder_inc = DoubleConvBlock(self.n_channels, 24)
    self.encoder_down1 = DownBlock(24, 24 * 2)
    self.encoder_down1_norm = nn.InstanceNorm2d(24 * 2, affine=True)
    self.encoder_down2 = DownBlock(24 * 2, 24 * 2 * 2)
    self.encoder_down3 = DownBlock(24 * 2 * 2, 24 * 2 * 2 * 2)
    self.encoder_down3_norm = nn.InstanceNorm2d(24 * 2 * 2 * 2, affine=True)
    self.encoder_bridge_down = BridgeDown(24 * 2 * 2 * 2, 24 * 2 * 2 * 2 * 2)
    self.decoder_bridge_up = BridgeUP(24 * 2 * 2 * 2 * 2, 24 * 2 * 2 * 2)
    self.decoder_up1 = UpBlock(24 * 2 * 2 * 2, 24 * 2 * 2)
    self.decoder_up2 = UpBlock(24 * 2 * 2, 24 * 2)
    self.decoder_up3 = UpBlock(24 * 2, 24)
    self.decoder_out = OutputBlock(24, self.n_channels)

  def forward(self, x):
    x1 = self.encoder_inc(x)
    x2 = self.encoder_down1(x1)
    x3 = self.encoder_down1_norm(x2)
    x3 = self.encoder_down2(x3)
    x4 = self.encoder_down3(x3)
    x5 = self.encoder_down3_norm(x4)
    x5 = self.encoder_bridge_down(x5)
    x = self.decoder_bridge_up(x5)
    x = self.decoder_up1(x, x4)
    x = self.decoder_up2(x, x3)
    x = self.decoder_up3(x, x2)
    out = self.decoder_out(x, x1)
    return out


class AWBNet_wo_R2(nn.Module):
  def __init__(self):
    super(AWBNet_wo_R2, self).__init__()
    self.global_net = globalSubNet()


  def forward(self, x, histogram):
    m, _ = self.global_net(histogram)
    m = torch.reshape(m, (x.size(0), 9, 3))
    # multiply
    y = x.clone()
    for i in range(m.size(0)):
      temp = torch.mm(self.kernel(torch.reshape(torch.squeeze(
        x[i, :, :, :]), (-1, 3))), torch.squeeze(m[i, :, :]))
      y[i, :, :, :] = torch.reshape(temp, (x.size(1), x.size(2),
                                           x.size(3)))
    return y


  @staticmethod
  def kernel(x):
    return torch.cat((x, x * x, torch.unsqueeze(x[:, 0] * x[:, 1], dim=-1),
                      torch.unsqueeze(x[:, 0] * x[:, 2], dim=-1),
                      torch.unsqueeze(x[:, 1] * x[:, 2], dim=-1)), dim=1)
