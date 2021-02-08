import torch

from classes.modules.common.BaseModel import BaseModel
from classes.modules.multiframe.rccnet.RCCNet import RCCNet


class ModelRCCNet(BaseModel):

    def __init__(self, input_size: int = 256, hidden_size: int = 128, backbone_type: str = "alexnet"):
        super().__init__()
        self._network = RCCNet(input_size, hidden_size, backbone_type).to(self._device)

    def predict(self, sequence: torch.Tensor, mimic: torch.Tensor = None) -> torch.Tensor:
        return self._network(sequence, mimic)

    def compute_loss(self, sequence: torch.Tensor, label: torch.Tensor, mimic: torch.Tensor = None) -> float:
        pred = self.predict(sequence, mimic)
        loss = self.get_angular_loss(pred, label)
        loss.backward()
        return loss.item()
