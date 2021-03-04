import math
import os
from typing import Union, Tuple, List

import torch
from torch.nn.functional import normalize

from auxiliary.settings import DEVICE


class BaseModel:

    def __init__(self):
        self._device = DEVICE
        self._network = None
        self.__optimizer = None

    def predict(self, image: torch.Tensor) -> Union[torch.Tensor, Tuple]:
        pass

    def compute_loss(self, img: torch.Tensor, label: torch.Tensor) -> Union[List, float]:
        pass

    def print_network(self):
        print(self._network)

    def log_network(self, path_to_log: str):
        open(os.path.join(path_to_log, "network.txt"), 'a+').write(str(self._network))

    def train_mode(self):
        self._network = self._network.train()

    def evaluation_mode(self):
        self._network = self._network.eval()

    def save(self, path_to_file: str):
        torch.save(self._network.state_dict(), path_to_file)

    def load(self, path_to_pretrained: str):
        self._network.load_state_dict(torch.load(path_to_pretrained, map_location=self._device))

    def set_optimizer(self, learning_rate: float, optimizer_type: str = "rmsprop"):
        optimizers_map = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop}
        self.__optimizer = optimizers_map[optimizer_type](self._network.parameters(), lr=learning_rate)

    def reset_gradient(self):
        self.__optimizer.zero_grad()

    def optimize(self):
        self.__optimizer.step()

    @staticmethod
    def get_angular_loss(pred: torch.Tensor, label: torch.Tensor, safe_v: float = 0.999999) -> torch.Tensor:
        dot = torch.clamp(torch.sum(normalize(pred, dim=1) * normalize(label, dim=1), dim=1), -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)
        return torch.mean(angle)
