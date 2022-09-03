import numpy as np
import torch
from torchvision import transforms
import utilities.utils as utls
import cv2


def image_wb(image=None, histogram=None, net_awb=None, device='cpu'):
  # check image size
  if image is not None:
    w, h, _ = image.shape
    if w % 2 ** 4 == 0:
      new_size_w = w
    else:
      new_size_w = w + 2 ** 4 - w % 2 ** 4

    if h % 2 ** 4 == 0:
      new_size_h = h
    else:
      new_size_h = h + 2 ** 4 - h % 2 ** 4

    inSz = (new_size_h, new_size_w)
    if not ((w, h) == inSz):
      image = cv2.resize(image, inSz, interpolation=cv2.INTER_NEAREST)


    image = np.array(image)
    assert len(image.shape) == 3, ('Input image should be 3 channels colored '
                                   'images')
    # HWC to CHW
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)

  if histogram is not None:
    histogram = histogram.transpose((2, 0, 1))
    histogram = torch.from_numpy(histogram)
    histogram = histogram.unsqueeze(0)
    histogram = histogram.to(device=device, dtype=torch.float32)

  net_awb.eval()
  with torch.no_grad():
    if image is not None and histogram is not None:
      output_awb, _ = net_awb(image, histogram)
    elif image is not None:
      output_awb = net_awb(image)
    elif histogram is not None:
      output_awb = net_awb(histogram)
    else:
      raise AssertionError('No input is given')
    output_awb = utls.from_tensor_to_image(output_awb)
    output_awb = cv2.resize(output_awb, (h, w), interpolation=cv2.INTER_NEAREST)
    output_awb = utls.outOfGamutClipping(output_awb)

    return output_awb * 255
