import numpy as np
import torch
import torch.utils.data as data

from classes.data.DataAugmenter import DataAugmenter


class BaseTemporalDataset(data.Dataset):

    def __init__(self, mode, input_size):
        self.__input_size = input_size
        self.__da = DataAugmenter(input_size)
        self._mode = mode
        self._data_dir, self._label_dir = "ndata_seq", "nlabel"
        self._paths_to_items = []

    def __getitem__(self, index: int) -> tuple:
        path_to_sequence = self._paths_to_items[index]
        label_path = path_to_sequence.replace(self._data_dir, self._label_dir)

        seq = np.array(np.load(path_to_sequence), dtype='float32')
        illuminant = np.array(np.load(label_path), dtype='float32')
        mimic = torch.from_numpy(self.__da.augment_mimic(seq).transpose((0, 3, 1, 2)).copy())

        if self._mode == "train":
            seq, color_bias = self.__da.augment_sequence(seq, illuminant)
            color_bias = np.array([[[color_bias[0][0], color_bias[1][1], color_bias[2][2]]]], dtype=np.float32)
            mimic = torch.mul(mimic, torch.from_numpy(color_bias).view(1, 3, 1, 1))
        else:
            seq = self.__da.resize_sequence(seq)

        seq = np.clip(seq, 0.0, 255.0) * (1.0 / 255)
        seq = self.__da.hwc_chw(self.__da.gamma_correct(self.__da.brg_to_rgb(seq)))

        seq = torch.from_numpy(seq.copy())
        illuminant = torch.from_numpy(illuminant.copy())

        return seq, mimic, illuminant, path_to_sequence

    def __len__(self) -> int:
        return len(self._paths_to_items)
