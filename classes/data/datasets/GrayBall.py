from __future__ import print_function

import glob
import os

from classes.data.datasets.BaseTemporalDataset import BaseTemporalDataset


class GrayBall(BaseTemporalDataset):

    def __init__(self, mode: str = "train", input_size: tuple = (224, 224), fold_num: int = 0):
        super().__init__(mode, input_size)
        path_to_dataset = os.path.join("dataset", "grayball", "preprocessed")
        scenes = sorted(os.listdir(path_to_dataset))
        test_scene = scenes.pop(fold_num)

        if self.__mode == "train":
            for scene in scenes:
                path_to_scene_data = os.path.join(path_to_dataset, scene, self._data_dir)
                self._paths_to_items += glob.glob(os.path.join(path_to_scene_data, "*.npy"))
        else:
            self._paths_to_items = glob.glob(os.path.join(path_to_dataset, test_scene, self._data_dir, "*.npy"))

        self._paths_to_items = sorted(self._paths_to_items)
