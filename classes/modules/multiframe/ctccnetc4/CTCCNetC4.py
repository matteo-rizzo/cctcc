from typing import Tuple

import torch
from torch import nn

from auxiliary.settings import DEVICE
from classes.modules.multiframe.tccnetc4.TCCNetC4 import TCCNetC4


class CTCCNetC4(nn.Module):

    def __init__(self):
        super().__init__()
        self.__device = DEVICE
        self.submodel1 = TCCNetC4()
        self.submodel2 = TCCNetC4()
        self.submodel3 = TCCNetC4()

    def __correct_sequence(self, seq: torch.Tensor, illuminant: torch.Tensor) -> torch.Tensor:
        # Linear to non-linear illuminant
        illuminant = illuminant.pow(1.0 / 2.2).unsqueeze(2).unsqueeze(3).unsqueeze(1)

        # Correct the image
        correction = (illuminant * torch.sqrt(torch.Tensor([3])).to(self.__device))
        correction = correction.expand(seq.shape[0], seq.shape[1], -1, -1, -1)
        corrected_seq = torch.div(seq, correction + 1e-10)

        # Normalize the image
        max_seq = torch.max(torch.max(torch.max(corrected_seq, dim=2)[0], dim=2)[0], dim=2)[0]
        normalized_seq = torch.div(corrected_seq, max_seq.unsqueeze(2).unsqueeze(2).unsqueeze(2) + 1e-10)

        return normalized_seq

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple:
        """
        @param a: the sequences of frames of shape "bs x ts x nc x h x w"
        @param b: the mimic sequences of shape "bs x ts x nc x h x w"
        @return: the normalized illuminant predictions from each step in the C4 cascade
        """
        o1 = self.submodel1(a, b)
        corrected_seq1_a = self.__correct_sequence(a, o1)
        corrected_seq1_b = self.__correct_sequence(b, o1)

        o2 = self.submodel2(corrected_seq1_a, corrected_seq1_b)
        corrected_seq2_a = self.__correct_sequence(a, torch.mul(o1, o2))
        corrected_seq2_b = self.__correct_sequence(b, torch.mul(o1, o2))

        o3 = self.submodel3(corrected_seq2_a, corrected_seq2_b)

        return o1, o2, o3
