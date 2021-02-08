import torch

from classes.modules.common.BaseModel import BaseModel
from classes.modules.multiframe.ctccnetc4.CTCCNetC4 import CTCCNetC4


class ModelCTCCNetC4(BaseModel):

    def __init__(self):
        super().__init__()
        self._network = CTCCNetC4().float().to(self._device)

    def predict(self, sequence: torch.Tensor, mimic: torch.Tensor = None) -> torch.Tensor:
        return self._network(sequence, mimic)

    def load_submodules(self, path_to_pretrained: str):
        self._network.submodel1.load_state_dict(torch.load(path_to_pretrained, map_location=self._device))
        self._network.submodel2.load_state_dict(torch.load(path_to_pretrained, map_location=self._device))
        self._network.submodel3.load_state_dict(torch.load(path_to_pretrained, map_location=self._device))

    @staticmethod
    def get_multiply_accumulated_loss(l1: torch.Tensor,
                                      l2: torch.Tensor,
                                      l3: torch.Tensor,
                                      a1: float = 0.33,
                                      a2: float = 0.33) -> torch.Tensor:
        return a1 * l1 + a2 * l2 + (1.0 - a1 - a2) * l3

    def compute_loss(self, o: list, y: torch.Tensor) -> tuple:
        l1 = self.get_angular_loss(o[0], y)
        l2 = self.get_angular_loss(torch.mul(o[0], o[1]), y)
        l3 = self.get_angular_loss(torch.mul(torch.mul(o[0], o[1]), o[2]), y)
        mal = self.get_multiply_accumulated_loss(l1, l2, l3)
        return l1, l2, l3, mal
