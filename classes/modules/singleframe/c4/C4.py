import torch

from auxiliary.settings import get_device
from classes.modules.singleframe.fc4.FC4 import FC4


class C4(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.submodel1 = FC4()
        self.submodel2 = FC4()
        self.submodel3 = FC4()

    @staticmethod
    def __correct_image(img: torch.Tensor, illuminant: torch.Tensor) -> torch.Tensor:
        # Correct the image
        nonlinear_illuminant = torch.pow(illuminant, 1.0 / 2.2)
        correction = nonlinear_illuminant.unsqueeze(2).unsqueeze(3) * torch.sqrt(torch.Tensor([3])).to(get_device())
        corrected_img = torch.div(img, correction + 1e-10)

        # Normalize the image
        max_img = torch.max(torch.max(torch.max(corrected_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
        max_img = max_img.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        return torch.div(corrected_img, max_img)

    def forward(self, x: torch.Tensor) -> tuple:
        """ x has shape [bs, 3, h, w] """

        o1 = self.submodel1(x)
        corrected_img1 = self.__correct_image(x, o1)

        o2 = self.submodel2(corrected_img1)
        corrected_img2 = self.__correct_image(x, torch.mul(o1, o2))

        o3 = self.submodel3(corrected_img2)

        return o1, o2, o3
