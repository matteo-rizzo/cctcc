import glob
import os
from typing import List, Tuple

import numpy as np
import torch

from classes.data.datasets.BaseTemporalDataset import BaseTemporalDataset


class GrayBall(BaseTemporalDataset):

    def __init__(self,
                 mode: str = "train",
                 input_size: Tuple = (224, 224),
                 fold: int = 0,
                 num_folds: int = 3,
                 return_labels: bool = False):

        super().__init__(mode, input_size)
        self.__return_labels = return_labels
        path_to_dataset = os.path.join("dataset", "grayball", "preprocessed")
        training_scenes = sorted(os.listdir(path_to_dataset))

        fold_size = len(training_scenes) // num_folds
        test_scenes = [training_scenes.pop(fold * fold_size) for _ in range(fold_size)]

        self.__scenes = training_scenes if self._mode == "train" else test_scenes
        for scene in self.__scenes:
            path_to_scene_data = os.path.join(path_to_dataset, scene, self._data_dir)
            self._paths_to_items += glob.glob(os.path.join(path_to_scene_data, "*.npy"))

        self._paths_to_items = sorted(self._paths_to_items)

    def get_scenes(self) -> List:
        return self.__scenes

    def __getitem__(self, index: int) -> Tuple:
        seq, mimic, illuminant, path_to_sequence = super().__getitem__(index)
        if self.__return_labels:
            labels_path = path_to_sequence.replace(self._data_dir, "nlabels")
            illuminants = np.array(np.load(labels_path), dtype='float32')
            illuminants = torch.from_numpy(illuminants.copy())
            return seq, mimic, illuminants, path_to_sequence
        return seq, mimic, illuminant, path_to_sequence
