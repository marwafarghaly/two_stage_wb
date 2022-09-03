from os.path import join
from os import path
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import random
from scipy.io import loadmat
import cv2
from utilities import utils as utils

class BasicDataset(Dataset):
  def __init__(self, imgs_dir, fold=0, patch_size=256, max_trdata=20000):
    self.imgs_dir = imgs_dir
    self.patch_size = patch_size
    # get selected training data based on the current fold
    if fold is not 0:
      tfolds = list(set([1, 2, 3]) - set([fold]))
      logging.info(
        f'Training process will use {max_trdata} training images randomly selected from folds {tfolds}')
      files = loadmat(join('folds', 'fold%d_.mat' % fold))
      files = files['training']
      self.imgfiles = []
      logging.info('Loading training images information...')
      for i in range(len(files)):
        temp_files = glob(imgs_dir + files[i][0][0])
        for file in temp_files:
          if file.endswith('.jpg'):
              self.imgfiles.append(file)
    elif fold is 0:
      logging.info(
        f'Training process will use {max_trdata} training images randomly selected from all training data')
      logging.info('Loading training images information...')
      self.imgfiles = [join(imgs_dir, file) for file in listdir(imgs_dir)
                       if file.endswith('.jpg') and not file.startswith('.')]
    else:
      logging.info(
        f'There is no fold {fold}! Training process will use all training data.')

    if max_trdata is not 0 and len(self.imgfiles) > max_trdata:
      random.shuffle(self.imgfiles)
      self.imgfiles = self.imgfiles[0:max_trdata]
    logging.info(f'Creating dataset with {len(self.imgfiles)} examples')

  def __len__(self):
    return len(self.imgfiles)

  @classmethod
  def preprocess(cls, img, patch_size, w, h, patch_coords, aug_op, scale=1,
                 input=False):
    if aug_op is 1:
      img = cv2.flip(img, 0)
    elif aug_op is 2:
      img = cv2.flip(img, 1)
    elif aug_op is 3:
      img = cv2.resize(img, (int(w * scale), int(h * scale)))

    img_nd = np.array(img)
    assert len(img_nd.shape) == 3, 'Training/validation images ' \
                                   'should be 3 channels colored images'
    img_patch = img_nd[patch_coords[1]:patch_coords[1] + patch_size,
             patch_coords[0]:patch_coords[0] + patch_size, :]
    # HWC to CHW
    img_patch = img_patch.transpose((2, 0, 1))
    return img_patch


  def __getitem__(self, i):
    gt_ext = 'G_AS.png'
    img_file = self.imgfiles[i]
    in_img = cv2.imread(img_file, -1)
    in_img = utils.from_bgr2rgb(in_img)  # convert from BGR to RGB
    in_img = utils.im2double(in_img, uint8=True)  # convert to double


    if path.exists(path.splitext(img_file)[0] + '_histogram.npy'):
      histogram = np.load(path.splitext(img_file)[0] +
                          '_histogram.npy', allow_pickle=False)
    else:
      histogram = utils.rgb_uv_hist(in_img, 64)
      np.save(path.splitext(img_file)[0] + '_histogram.npy', histogram)

    histogram = histogram.transpose((2, 0, 1))

    # get image size
    h, w, _ = in_img.shape
    # get ground truth images
    parts = img_file.split('_')
    base_name = ''
    for i in range(len(parts) - 2):
      base_name = base_name + parts[i] + '_'
    gt_awb_file = base_name + gt_ext
    awb_img = cv2.imread(gt_awb_file, -1)
    awb_img = utils.from_bgr2rgb(awb_img)
    awb_img = utils.im2double(awb_img, uint8=True)
    # get flipping option
    aug_op = np.random.randint(4)
    if aug_op == 3:
      scale = np.random.uniform(low=1.0, high=1.2)
    else:
      scale = 1
    # get random patch coord
    awb_img_patches = 0
    C = 0
    while np.mean(awb_img_patches) < 0.005 and C < 10:
      patch_x = np.random.randint(0, high=w - self.patch_size)
      patch_y = np.random.randint(0, high=h - self.patch_size)
      in_img_patches = self.preprocess(in_img, self.patch_size, w, h,
                                       (patch_x, patch_y), aug_op, scale=scale)
      awb_img_patches = self.preprocess(awb_img, self.patch_size, w, h,
                                        (patch_x, patch_y), aug_op, scale=scale)
      C = C + 1

    return {'image': torch.from_numpy(in_img_patches),
            'gt_AWB': torch.from_numpy(awb_img_patches),
            'histogram': torch.from_numpy(histogram)
            }

  @staticmethod
  def compute_hist(img_dir):
    logging.info(f'Computing histogram features offline in {img_dir}.')
    logging.info('Loading training images information...')
    imgfiles = [join(img_dir, file) for file in listdir(img_dir)
                     if file.endswith('.jpg') and not file.startswith('.')]

    count = 0
    for img_file in imgfiles:
      if count % 100 == 0:
        logging.info(f'processing ({count}/{len(imgfiles)}) ...')
      count = count + 1
      if path.exists(path.splitext(img_file)[0] + '_histogram.npy'):
        histogram = np.load(path.splitext(img_file)[0] +
                            '_histogram.npy', allow_pickle=False)
      else:
        in_img = cv2.imread(img_file, -1)
        in_img = utils.from_bgr2rgb(in_img)  # convert from BGR to RGB
        in_img = utils.im2double(in_img, uint8=True)  # convert to double

        histogram = utils.rgb_uv_hist(in_img, 64)
        np.save(path.splitext(img_file)[0] + '_histogram.npy', histogram)

