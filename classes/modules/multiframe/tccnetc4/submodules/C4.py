import torch
from torch import nn

from auxiliary.settings import DEVICE
from classes.modules.common.squeezenet.SqueezeNetLoader import SqueezeNetLoader
from classes.modules.singleframe.fc4.FC4 import FC4


class C4(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.__device = DEVICE
        self.submodel1 = FC4()
        self.submodel2 = FC4()
        self.submodel3 = nn.Sequential(*list(SqueezeNetLoader().load(pretrained=True).children())[0][:12])

    def __correct_frames(self, frames: torch.Tensor, illuminant: torch.Tensor) -> torch.Tensor:
        # Correct the image
        nonlinear_illuminant = torch.pow(illuminant, 1.0 / 2.2).unsqueeze(2).unsqueeze(3)
        correction = nonlinear_illuminant * torch.sqrt(torch.Tensor([3])).to(self.__device)
        corrected_frames = torch.div(frames, correction + 1e-10)

        # Normalize the image
        max_img = torch.max(torch.max(torch.max(corrected_frames, dim=1)[0], dim=1)[0], dim=1)[0]
        normalized_frames = torch.div(corrected_frames, max_img.unsqueeze(1).unsqueeze(1).unsqueeze(1) + 1e-10)
        return normalized_frames

    def forward(self, x: torch.Tensor) -> tuple:
        """ x has shape [bs, 3, h, w] """

        o1 = self.submodel1(x)
        corrected_img1 = self.__correct_frames(x, o1)

        o2 = self.submodel2(corrected_img1)
        corrected_img2 = self.__correct_frames(x, torch.mul(o1 + 1e-10, o2 + 1e-10))

        o3 = self.submodel3(corrected_img2)

        return o3
