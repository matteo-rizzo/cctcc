from typing import Tuple, List

import torch

from auxiliary.settings import NUM_STAGES
from classes.modules.common.BaseModel import BaseModel
from classes.modules.multiframe.ctccnet2.CTCCNet2 import CTCCNet2


class ModelCTCCNet2(BaseModel):

    def __init__(self):
        super().__init__()
        self._network = CTCCNet2().float().to(self._device)

    def predict(self, seq_temp: torch.Tensor, seq_shot: torch.Tensor = None) -> List:
        return self._network(seq_temp, seq_shot)

    @staticmethod
    def get_multiply_accumulated_loss(l1: torch.Tensor,
                                      l2: torch.Tensor,
                                      l3: torch.Tensor,
                                      a1: float = 0.33,
                                      a2: float = 0.33) -> torch.Tensor:
        return a1 * l1 + a2 * l2 + (1.0 - a1 - a2) * l3

    def compute_loss(self, o: List, y: torch.Tensor) -> Tuple:
        self.reset_gradient()
        stages_loss, mal = self.get_loss(o, y)
        mal.backward()
        self.optimize()
        return stages_loss, mal

    def get_loss(self, o: List, y: torch.Tensor) -> Tuple:
        stage_out, stages_loss = None, []
        for stage in NUM_STAGES:
            stage_out = torch.mul(stage_out, o[stage]) if stage - 1 > 0 else o[stage]
            stages_loss.append(self.get_angular_loss(stage_out, y))
        mal = sum(stages_loss)
        return stages_loss, mal
