import numpy as np
import torch
import cv2


def outOfGamutClipping(I):
  """ Clips out-of-gamut pixels. """
  I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
  I[I < 0] = 0  # any pixel is below 0, clip it to 0
  return I

def from_tensor_to_image(tensor, device='cuda'):
  """ converts tensor to image """
  tensor = torch.squeeze(tensor, dim=0)
  if device == 'cpu':
    image = tensor.data.numpy()
  else:
    image = tensor.cpu().data.numpy()
  # CHW to HWC
  image = image.transpose((1, 2, 0))
  image = from_rgb2bgr(image)
  return image

def rgb_uv_hist(I, h):
  """ Computes an RGB-uv histogram tensor. """
  sz = np.shape(I)  # get size of current image
  if sz[0] * sz[1] > 202500:  # resize if it is larger than 450*450
    factor = np.sqrt(202500 / (sz[0] * sz[1]))  # rescale factor
    newH = int(np.floor(sz[0] * factor))
    newW = int(np.floor(sz[1] * factor))
    I = cv2.resize(I, (newW, newH), interpolation=cv2.INTER_NEAREST)
  I_reshaped = I[(I > 0).all(axis=2)]
  eps = 6.4 / h
  hist = np.zeros((h, h, 3))  # histogram will be stored here
  Iy = np.linalg.norm(I_reshaped, axis=1)  # intensity vector
  for i in range(3):  # for each histogram layer, do
    r = []  # excluded channels will be stored here
    for j in range(3):  # for each color channel do
      if j != i:
        r.append(j)
    Iu = np.log(I_reshaped[:, i] / I_reshaped[:, r[1]])
    Iv = np.log(I_reshaped[:, i] / I_reshaped[:, r[0]])
    hist[:, :, i], _, _ = np.histogram2d(
      Iu, Iv, bins=h, range=((-3.2 - eps / 2, 3.2 - eps / 2),) * 2,
      weights=Iy)
    norm_ = hist[:, :, i].sum()
    hist[:, :, i] = np.sqrt(hist[:, :, i] / norm_)
  return hist

#
# def from_image_to_tensor(image):
#   image = from_bgr2rgb(image)
#   image = im2double(image)  # convert to double
#   image = np.array(image)
#   assert len(image.shape) == 3, ('Input image should be 3 channels colored '
#                                  'images')
#   # HWC to CHW
#   image = image.transpose((2, 0, 1))
#   return torch.unsqueeze(torch.from_numpy(image), dim=0)


def from_bgr2rgb(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB


def from_rgb2bgr(image):
  return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert from BGR to RGB

def im2double(im, uint8=False):
  """ Returns a double image [0,1] of the uint im. """
  if uint8 == True:
    return im.astype('float')/255
  else:
    return im.astype('float') / 65535



