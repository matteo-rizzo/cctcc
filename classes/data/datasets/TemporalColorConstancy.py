from __future__ import print_function

import glob
import os

import numpy as np
import torch
import torch.utils.data as data

from classes.data.DataAugmenter import DataAugmenter


class TemporalColorConstancy(data.Dataset):

    def __init__(self, mode="train", input_size=(224, 224), data_folder="tcc_split"):

        self.__data_dir = "ndata_seq"
        self.__label_dir = "nlabel"
        path_to_dataset = os.path.join("dataset", "tcc", "preprocessed", data_folder)
        path_to_data = os.path.join(path_to_dataset, self.__data_dir)

        self.__mode = mode
        self.__input_size = input_size
        self.__da = DataAugmenter(input_size)

        self.__paths_to_items = glob.glob(os.path.join(path_to_data, "{}*.npy".format(mode)))
        self.__paths_to_items.sort(key=lambda x: int(x.split(mode)[-1][:-4]))

    def __getitem__(self, index: int) -> tuple:
        path_to_sequence = self.__paths_to_items[index]
        label_path = path_to_sequence.replace(self.__data_dir, self.__label_dir)

        img = np.array(np.load(path_to_sequence), dtype='float32')
        illuminant = np.array(np.load(label_path), dtype='float32')
        mimic = torch.from_numpy(self.__da.augment_mimic(img).transpose((0, 3, 1, 2)).copy())

        if self.__mode == "train":
            img, color_bias = self.__da.augment_sequence(img, illuminant)
            color_bias = np.array([[[color_bias[0][0], color_bias[1][1], color_bias[2][2]]]], dtype=np.float32)
            mimic = torch.mul(mimic, torch.from_numpy(color_bias).view(1, 3, 1, 1))
        else:
            img = self.__da.resize_sequence(img)

        img = np.clip(img, 0.0, 255.0) * (1.0 / 255)
        img = self.__da.hwc_chw(self.__da.gamma_correct(self.__da.brg_to_rgb(img)))

        img = torch.from_numpy(img.copy())
        illuminant = torch.from_numpy(illuminant.copy())

        return img, mimic, illuminant, path_to_sequence

    def __len__(self) -> int:
        return len(self.__paths_to_items)
