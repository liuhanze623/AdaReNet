import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

import os
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def load_dataset_my(root_dir, redux, params, shuffled=False, single=False):
    """Loads dataset and returns corresponding data loader."""

    # Create Torch dataset
    noise = (params.noise_type, params.noise_param)

    # Instantiate appropriate dataset class
    if params.noise_type == 'mc':
        dataset = MonteCarloDataset(root_dir, redux, params.crop_size,
            clean_targets=params.clean_targets)
    else:
        dataset = NoisyDataset(root_dir, redux, params.crop_size,
            clean_targets=params.clean_targets, noise_dist=noise, seed=params.seed)                    
       

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, redux=0, crop_size=128, clean_targets=False):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size
        self.clean_targets = clean_targets

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs
    


    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')


    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)


class NoisyDataset(AbstractDataset):
    """Class for injecting random noise into dataset."""

    def __init__(self, root_dir, redux, crop_size, clean_targets=False,                         
        noise_dist=('gaussian', 50.), seed=None):
    
        """Initializes noisy image dataset."""

        super(NoisyDataset, self).__init__(root_dir, redux, crop_size, clean_targets)                    
        
        self.imgs = os.listdir(root_dir)
        if redux:
            self.imgs = self.imgs[:redux]

        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        


    def _add_noise(self, img):
        """Adds Gaussian or Poisson noise to image."""

        w, h = img.size
        c = len(img.getbands())

       
        if self.noise_type == 'poisson':
            noise = np.random.poisson(img)
            noise_img = img + noise
            noise_img = 255 * (noise_img / np.amax(noise_img))                               
        else:
            if self.seed:
                std = 25                                     
            else:
                std = np.random.uniform(0, self.noise_param)
            noise = np.random.normal(0, std, (h, w, c))

            noise_img = np.array(img) + noise

        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noise_img)


    


    def _rotate(self, img):                                                                                               
        """Rotates the image by a random angle."""
        rotation_angle = random.randint(0, 3) * 90  
        img = np.rot90(img, k=rotation_angle // 90)
        return Image.fromarray(img), Image.fromarray(img)
    
    
    def _corrupt(self, img):
        """Corrupts images (Gaussian, Poisson, or text overlay)."""

        if self.noise_type in ['gaussian', 'poisson']:
            return self._add_noise(img)
        elif self.noise_type == 'text':
            return self._add_text_overlay(img)
        else:
            raise ValueError('Invalid noise type: {}'.format(self.noise_type))


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        img_path = os.path.join(self.root_dir, self.imgs[index])
        img =  Image.open(img_path).convert('RGB')

        # Random square crop
        if self.crop_size != 0:
            img = self._random_crop([img])[0]    
            

        
        tmp = self._corrupt(img)                  
       
        source = tvF.to_tensor(self._corrupt(img))

       
        if self.clean_targets:
            target = tvF.to_tensor(img)
        else:
            target = tvF.to_tensor(img)
            

        return source, target