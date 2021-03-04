import glob
import os
from typing import List, Tuple

from classes.data.datasets.BaseTemporalDataset import BaseTemporalDataset


class GrayBall(BaseTemporalDataset):

    def __init__(self, mode: str = "train", input_size: Tuple = (224, 224), fold: int = 0, num_folds: int = 3):
        super().__init__(mode, input_size)
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
