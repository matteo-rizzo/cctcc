from typing import Tuple, List

import torch

from auxiliary.settings import NUM_STAGES
from classes.modules.common.BaseModel import BaseModel
from classes.modules.multiframe.ctccnet2.CTCCNet2 import CTCCNet2


class ModelCTCCNet2(BaseModel):

    def __init__(self):
        super().__init__()
        self._network = CTCCNet2().float().to(self._device)

    def predict(self, seq_temp: torch.Tensor, seq_shot: torch.Tensor = None, return_preds: bool = False) -> List:
        return self._network(seq_temp, seq_shot, return_preds)

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
        for stage in range(NUM_STAGES):
            stage_out = torch.mul(stage_out, o[stage]) if stage - 1 > 0 else o[stage]
            stages_loss.append(self.get_angular_loss(stage_out, y))
        mal = sum(stages_loss)
        return stages_loss, mal

    def compute_corr_loss(self, o: List, y: torch.Tensor) -> Tuple:
        self.reset_gradient()
        cas_loss, cas_mal, cor_loss, cor_mal = self.get_corr_loss(o, y)
        mal = cas_mal + cor_mal
        mal.backward()
        self.optimize()
        return cas_loss, cas_mal, cor_loss, cor_mal

    def get_corr_loss(self, o: List, y: torch.Tensor) -> Tuple:
        outputs, preds = zip(*o)
        cas_out, cor_out, cas_loss, cor_loss = None, None, [], []
        for stage in range(NUM_STAGES):
            cas_out = torch.mul(cas_out, outputs[stage]) if stage - 1 > 0 else outputs[stage]
            cas_loss.append(self.get_angular_loss(cas_out, y[:, -1, :]))
            cor_out = torch.mul(cor_out, preds[stage]) if stage - 1 > 0 else preds[stage]
            cor_loss.append(self.get_angular_loss(cor_out.permute(1, 0, 2), y))
        cas_mal, cor_mal = sum(cas_loss), sum(cor_loss)
        return cas_loss, cas_mal, cor_loss, cor_mal
