import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from auxiliary.settings import get_device


def correct(img: np.array, illuminant: torch.Tensor) -> Image:
    """
    Corrects the color of the illuminant of a linear image based on an estimated (linear) illuminant
    @param img: a linear image
    @param illuminant: a linear illuminant
    @return: a non-linear color-corrected version of the input image
    """
    img = TF.to_tensor(img)

    # Correct the image
    correction = illuminant.unsqueeze(2).unsqueeze(3) * torch.sqrt(torch.Tensor([3])).to(get_device())
    corrected_img = torch.div(img, correction + 1e-10)

    # Normalize the image
    max_img = torch.max(torch.max(torch.max(corrected_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
    max_img = max_img.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    normalized_img = torch.div(corrected_img, max_img)

    linear_image = torch.pow(normalized_img, 1.0 / 2.2)
    return TF.to_pil_image(linear_image.squeeze(), mode="RGB")


def linear_to_nonlinear(img: Image) -> Image:
    return TF.to_pil_image(torch.pow(TF.to_tensor(img), 1.0 / 2.2).squeeze(), mode="RGB")
