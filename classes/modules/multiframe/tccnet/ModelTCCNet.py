import torch

from classes.modules.common.BaseModel import BaseModel
from classes.modules.multiframe.tccnet.TCCNet import TCCNet


class ModelTCCNet(BaseModel):

    def __init__(self, device: torch.device, hidden_size: int = 128, kernel_size: int = 5):
        super().__init__(device)
        self._network = TCCNet(hidden_size, kernel_size).to(self._device)

    def predict(self, sequence: torch.Tensor, mimic: torch.Tensor = None) -> torch.Tensor:
        return self._network(sequence, mimic)

    def compute_loss(self, sequence: torch.Tensor, label: torch.Tensor, mimic: torch.Tensor = None) -> float:
        pred = self.predict(sequence, mimic)
        loss = self.get_angular_loss(pred, label)
        loss.backward()
        return loss.item()
