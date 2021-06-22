#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by andy

import cv2
import json
import numpy as np
import os
import io
import random
import scipy.io
import sys
import torch.utils.data.dataset

from datasets.image_folder import make_dataset
from config import cfg
from datetime import datetime as dt
from enum import Enum, unique
from utils.imgio_gen import readgen
import utils.network_utils

class DatasetType(Enum):
    TRAIN = 0
    TEST  = 1

class GoProDataset(torch.utils.data.dataset.Dataset):
    """GoProDataset class used for PyTorch DataLoader"""

    def __init__(self, file_list_with_metadata, transforms=None):
        self.file_list = file_list_with_metadata
        self.transforms = transforms
        print("Andy: processing GOPRO dataset\n")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name, imgs, imgs2 = self.get_datum(idx)
        imgs, imgs2 = self.transforms(imgs,imgs2)
        #print(imgs[2].size())
        # imgs[2]=torch.mean(imgs[2], 0,keepdim=True)
        #print(imgs[2].size())
        return name, imgs, imgs2

    def get_datum(self, idx):

        name = self.file_list[idx]['name']
        img_blur_path = self.file_list[idx]['img_blur']
        img_shape_path = self.file_list[idx]['img_shape']
        img_edge_path = self.file_list[idx]['img_edge']
        # print(img_blur_path)
        # print(img_shape_path)
        # print(img_edge_path)
        img_blur = readgen(img_blur_path).astype(np.float32)
        img_shape = readgen(img_shape_path).astype(np.float32)
        img_edge = readgen(img_edge_path).astype(np.float32)
        # print(img_blur)
        # print(img_edge.shape)
        imgs = [img_blur, img_shape]
        imgs2 = [img_edge]

        # cv2.imshow(img_shape)
        # cv2.waitKey(0)
        return name, imgs, imgs2
# //////////////////////////////// = End of GoProDataset Class Definition = ///////////////////////////////// #

class GoProLoader:
    def __init__(self):
        print("Andy: processing GOPRO dataloader\n")
        self.img_train_blur_path_template  = cfg.DIR.Train_Blur_image_PATH
        self.img_train_shape_path_template = cfg.DIR.Train_Shape_image_PATH
        self.img_train_edge_path_template  = cfg.DIR.Train_Edge_image_PATH
        self.img_test_blur_path_template   = cfg.DIR.Test_Blur_image_PATH
        self.img_test_shape_path_template  = cfg.DIR.Test_Shape_image_PATH
        self.img_test_edge_path_template   = cfg.DIR.Test_Blur_image_PATH

    def get_dataset(self, dataset_type, transforms=None):
        files = []
        # Load data for each sequence
        if dataset_type == DatasetType.TRAIN:
            paths, names = make_dataset(self.img_train_blur_path_template)
            for name in names:
                img_blur_path = self.img_train_blur_path_template + "/" + name
                img_shape_path = self.img_train_shape_path_template + "/" + name
                img_edge_path = self.img_train_edge_path_template + "/" + name
                files.append({
                    'name': name,
                    'img_blur': img_blur_path,
                    'img_shape': img_shape_path,
                    'img_edge': img_edge_path,
                })
        if dataset_type == DatasetType.TEST:
            paths, names = make_dataset(self.img_test_blur_path_template)
            for name in names:
                img_blur_path = self.img_test_blur_path_template + "/" + name
                img_shape_path = self.img_test_shape_path_template + "/" + name
                img_edge_path = self.img_test_edge_path_template + "/" + name
                files.append({
                    'name': name,
                    'img_blur': img_blur_path,
                    'img_shape': img_shape_path,
                    'img_edge': img_edge_path,
                })


        print('[INFO] %s Complete collecting files of the dataset for %s. Total Pair Numbur: %d.\n' % (dt.now(), dataset_type.name, len(files)))
        return GoProDataset(files, transforms)


# /////////////////////////////// = End of GoProLoader Class Definition = /////////////////////////////// #
class predictDataset(torch.utils.data.dataset.Dataset):
    """predictDataset class used for PyTorch DataLoader"""

    def __init__(self, file_list_with_metadata, transforms=None):
        self.file_list = file_list_with_metadata
        self.transforms = transforms
        print("Andy: processing predict dataset\n")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name, imgs = self.get_datum(idx)
        imgs = self.transforms(imgs)
        #print(imgs[2].size())
        #print(imgs[2].size())
        return name, imgs

    def get_datum(self, idx):

        name = self.file_list[idx]['name']
        img_blur_path = self.file_list[idx]['img_blur']

        img_blur = readgen(img_blur_path).astype(np.float32)
        #img_edge = cv2.cvtColor(img_edge, cv2.COLOR_RGB2GRAY).astype(np.float32)
        imgs = [img_blur, img_blur]

        # cv2.imshow(img_shape)
        # cv2.waitKey(0)
        return name, imgs
# //////////////////////////////// = End of KohlerDataset Class Definition = ///////////////////////////////// #

class predictLoader:
    def __init__(self):
        print("Andy: processing predict dataloader\n")
        self.img_test_blur_path_template   = cfg.DIR.Predict_image_PATH


    def get_dataset(self, dataset_type, transforms=None):
        files = []
        # Load data for each sequence

        paths, names = make_dataset(self.img_test_blur_path_template)
        for name in names:
            img_blur_path = self.img_test_blur_path_template + "/" + name
            files.append({
                'name': name,
                'img_blur': img_blur_path,
            })

        print('[INFO] %s Complete collecting files of the dataset for %s. Total Pair Numbur: %d.\n' % (dt.now(), dataset_type.name, len(files)))
        return predictDataset(files, transforms)


# /////////////////////////////// = End of KohlerLoader Class Definition = /////////////////////////////// #



DATASET_LOADER_MAPPING = {
    'GoPro': GoProLoader,
    'predict': predictLoader,
}
