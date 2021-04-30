from typing import Union

import torch
from torch import nn

from auxiliary.settings import DEVICE, NUM_STAGES
from classes.modules.multiframe.ctccnet2.submodules.TCCNet import TCCNet


class CTCCNet2(nn.Module):

    def __init__(self):
        super().__init__()
        self.__device = DEVICE
        self.submodules = nn.ModuleList()
        for _ in range(NUM_STAGES):
            self.submodules.append(TCCNet())

    def __correct_sequence(self, seq: torch.Tensor, illuminants: torch.Tensor) -> torch.Tensor:
        """ Correct each frame in the sequence using its predicted illuminant """

        # Linear to non-linear illuminant
        illuminants = illuminants.pow(1.0 / 2.2).permute(1, 0, 2).unsqueeze(3).unsqueeze(4)

        # Correct the image
        correction = (illuminants * torch.sqrt(torch.Tensor([3])).to(self.__device))
        corrected_seq = torch.div(seq, correction + 1e-10)

        # Normalize the image
        max_seq = torch.max(torch.max(torch.max(corrected_seq, dim=2)[0], dim=2)[0], dim=2)[0]
        normalized_seq = torch.div(corrected_seq, max_seq.unsqueeze(2).unsqueeze(2).unsqueeze(2) + 1e-10)

        return normalized_seq

    def forward(self,
                seq_temp: torch.Tensor,
                seq_shot: torch.Tensor,
                return_preds: bool = False) -> Union[torch.Tensor, list]:
        """
        @param seq_temp: the sequences of frames of shape "bs x ts x nc x h x w"
        @param seq_shot: the mimic sequences of shape "bs x ts x nc x h x w"
        @return: the normalized illuminant predictions from each step in the cascade
        """

        # Iterate over each stage in the cascade
        outputs, preds = [], []

        for submodule in self.submodules[:-1]:
            # Get the predictions for the current stage
            output, t_preds, s_preds = submodule(seq_temp, seq_shot)

            # Correct the temporal sequence
            seq_temp = self.__correct_sequence(seq_temp, t_preds)

            # Correct the shot frame sequence
            seq_shot = self.__correct_sequence(seq_shot, s_preds)

            # Add the predicted shot frame illuminant to the list of outputs to return
            outputs.append(output)
            preds.append(t_preds)

        # Get the predicted illuminant for the last stage of the cascade
        output, t_preds, _ = self.submodules[-1](seq_temp, seq_shot)
        outputs.append(output)
        preds.append(t_preds)

        if return_preds:
            return list(zip(outputs, preds))

        return outputs
