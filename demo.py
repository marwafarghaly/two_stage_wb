import argparse
import logging
import os
import torch
import cv2
import utilities.utils as utls
import utilities.image_wb as correction
from arch import awb_net
from os import path
import numpy as np


def get_args():
  parser = argparse.ArgumentParser(description='Image WB.')
  parser.add_argument('--model_dir', '-m', default='./models',
                      help="Specify the directory of the trained model.",
                      dest='model_dir')
  parser.add_argument('--input_dir', '-i', help='Input image directory',
                      dest='input_dir',
                      default='./data_set_2/Images/')
  parser.add_argument('--output_dir', '-o',
                      default='result_images_set_2',
                      help='Directory to save the output images',
                      dest='out_dir')
  parser.add_argument('--save', '-s', action='store_true',
                      help="Save the output images",
                      default=True, dest='save')
  parser.add_argument('--device', '-d', default='cuda',
                      help="Device: cuda or cpu.", dest='device')

  return parser.parse_args()


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  args = get_args()
  if args.device.lower() == 'cuda':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  else:
    device = torch.device('cpu')

  input_dir = args.input_dir
  out_dir = args.out_dir
  tosave = args.save

  logging.info(f'Using device {device}')

  if tosave:
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)

  if os.path.exists(os.path.join(args.model_dir, 'net_awb.pth')):
    # load awb net
    net_awb = awb_net.AWBNet()
    logging.info(
      "Loading model {}".format(os.path.join(args.model_dir, 'net_awb.pth')))
    net_awb.to(device=device)
    net_awb.load_state_dict(
      torch.load(os.path.join(args.model_dir, 'net_awb.pth'),
                 map_location=device))
    net_awb.eval()
  else:
    raise Exception('Model not found!')

  imgfiles = []
  valid_images = (".jpg", ".png")
  for fn in os.listdir(input_dir):
    if fn.lower().endswith(valid_images):
      imgfiles.append(os.path.join(input_dir, fn))

  for fn in imgfiles:

    logging.info("Processing image {} ...".format(fn))
    in_img = cv2.imread(fn, -1)
    in_img = utls.from_bgr2rgb(in_img)  # convert from BGR to RGB
    in_img = utls.im2double(in_img, uint8=True)  # convert to double
    if path.exists(path.splitext(fn)[0] + '_histogram.npy'):
      histogram = np.load(path.splitext(fn)[0] + '_histogram.npy',
                          allow_pickle=False)
    else:
      histogram = utls.rgb_uv_hist(in_img, 64)
      np.save(path.splitext(fn)[0] + '_histogram.npy', histogram)

    _, fname = os.path.split(fn)
    name, _ = os.path.splitext(fname)
    out_awb = correction.image_wb(image=in_img, histogram=histogram,
                                  net_awb=net_awb, device=device)
    if tosave:
      #out_awb.save(os.path.join(out_dir, name + '.png'))
      cv2.imwrite(os.path.join(out_dir, name + '.png'), out_awb)
