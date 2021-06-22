#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Andy
'''ref: http://pytorch.org/docs/master/torchvision/transforms.html'''


import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from config import cfg
from PIL import Image
import random
import numbers
class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs, inputs2):
        for t in self.transforms:
            inputs,inputs2 = t(inputs, inputs2)
        return inputs,inputs2


class ColorJitter(object):
    def __init__(self, color_adjust_para):
        """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
        """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
        """saturation [max(0, 1 - saturation), 1 + saturation] or the given [min, max]"""
        """hue [-hue, hue] 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5"""
        '''Ajust brightness, contrast, saturation, hue'''
        '''Input: PIL Image, Output: PIL Image'''
        self.brightness, self.contrast, self.saturation, self.hue = color_adjust_para

    def __call__(self, inputs, inputs2):
        inputs = [Image.fromarray(np.uint8(inp)) for inp in inputs]
        inputs2 = [Image.fromarray(np.uint8(inp)) for inp in inputs2]
        if self.brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            inputs = [F.adjust_brightness(inp, brightness_factor) for inp in inputs]
            inputs2 = [F.adjust_brightness(inp, brightness_factor) for inp in inputs2]

        if self.contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            inputs = [F.adjust_contrast(inp, contrast_factor) for inp in inputs]
            inputs2 = [F.adjust_contrast(inp, contrast_factor) for inp in inputs2]

        if self.saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            inputs = [F.adjust_saturation(inp, saturation_factor) for inp in inputs]
            inputs2 = [F.adjust_saturation(inp, saturation_factor) for inp in inputs2]

        if self.hue > 0:
            hue_factor = np.random.uniform(-self.hue, self.hue)
            inputs = [F.adjust_hue(inp, hue_factor) for inp in inputs]
            inputs2 = [F.adjust_hue(inp, hue_factor) for inp in inputs2]

        inputs = [np.asarray(inp) for inp in inputs]
        inputs = [inp.clip(0,255) for inp in inputs]

        inputs2 = [np.asarray(inp) for inp in inputs2]
        inputs2 = [inp.clip(0,255) for inp in inputs2]

        return inputs, inputs2

class RandomColorChannel(object):
    def __call__(self, inputs, inputs2):
        random_order = np.random.permutation(3)
        inputs = [inp[:,:,random_order] for inp in inputs]
        inputs2 = [inp[:,:,random_order] for inp in inputs2]
        return inputs, inputs2

class RandomGaussianNoise(object):
    def __init__(self, gaussian_para):
        self.mu = gaussian_para[0]
        self.std_var = gaussian_para[1]

    def __call__(self, inputs, inputs2):

        shape = inputs[0].shape
        gaussian_noise = np.random.normal(self.mu, self.std_var, shape)
        # only apply to blurry images
        inputs[0] = inputs[0]+gaussian_noise
        # inputs2[0] = inputs2[0]+gaussian_noise

        inputs = [inp.clip(0, 1) for inp in inputs]
        inputs2 = [inp.clip(0, 1) for inp in inputs2]

        return inputs, inputs2

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
    def __call__(self, inputs, inputs2):
        assert(all([isinstance(inp, np.ndarray) for inp in inputs]))
        inputs = [inp/self.std -self.mean for inp in inputs]
        inputs2 = [inp/self.std -self.mean for inp in inputs2]
        return inputs, inputs2

class CenterCrop(object):

    def __init__(self, crop_size):
        """Set the height and weight before and after cropping"""

        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, inputs):
        input_size_h, input_size_w, _ = inputs[0].shape
        x_start = int(round((input_size_w - self.crop_size_w) / 2.))
        y_start = int(round((input_size_h - self.crop_size_h) / 2.))

        inputs = [inp[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for inp in inputs]

        return inputs

class RandomDownsample(object):

    def __call__(self, inputs):
        if random.random() < 0.5:
            input_size_h, input_size_w, _ = inputs[0].shape
            output_size_h = input_size_h * 0.6
            output_size_w = input_size_w * 0.6
            size = (int(output_size_w), int(output_size_h))
            #shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

            inputs = [cv2.resize(inp, size, interpolation=cv2.INTER_AREA) for inp in inputs]
        return inputs

class RandomCrop(object):

    def __init__(self, crop_size):
        """Set the height and weight before and after cropping"""
        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, inputs, inputs2):
        input_size_h, input_size_w, _ = inputs[0].shape
        x_start = random.randint(0, input_size_w - self.crop_size_w)
        y_start = random.randint(0, input_size_h - self.crop_size_h)
        inputs = [inp[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for inp in inputs]
        # x_start2 = x_start*2
        # y_start2 = y_start*2
        # inputs2 = [inp[y_start2: y_start2 + self.crop_size_h*2, x_start2: x_start2 + self.crop_size_w*2] for inp in inputs2]

        inputs2 = [inp[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for inp in inputs2]
        return inputs, inputs2

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5 left-right"""

    def __call__(self, inputs, inputs2):
        if random.random() < 0.5:
            '''Change the order of 0 and 1, for keeping the net search direction'''
            # inputs[0] = np.copy(np.fliplr(inputs[0]))
            # inputs[1] = np.copy(np.fliplr(inputs[1]))
            # inputs2[0] = np.copy(np.fliplr(inputs2[0]))


        return inputs,inputs2


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5  up-down"""
    def __call__(self, inputs, inputs2):
        if random.random() < 0.5:
            inputs = [np.copy(np.flipud(inp)) for inp in inputs]
            inputs2 = [np.copy(np.flipud(inp)) for inp in inputs2]
        return inputs, inputs2


class RandomHorizonFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5  up-down"""
    def __call__(self, inputs, inputs2):
        if random.random() < 0.5:
            inputs = [np.copy(np.fliplr(inp)) for inp in inputs]
            inputs2 = [np.copy(np.fliplr(inp)) for inp in inputs2]
        return inputs, inputs2


class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, inputs, inputs2):
        assert(isinstance(inputs[0], np.ndarray) and isinstance(inputs[1], np.ndarray))
        inputs = [np.transpose(inp, (2, 0, 1)) for inp in inputs]
        inputs_tensor = [torch.from_numpy(inp).float() for inp in inputs]

        assert(isinstance(inputs2[0], np.ndarray))
        inputs2 = [np.transpose(inp, (2, 0, 1)) for inp in inputs2]
        inputs_tensor2 = [torch.from_numpy(inp).float() for inp in inputs2]
        return inputs_tensor,inputs_tensor2

