import torch
from torch import nn
from torch.nn.functional import normalize

from classes.modules.common.squeezenet.SqueezeNetLoader import SqueezeNetLoader


class FC4(torch.nn.Module):

    def __init__(self, squeezenet_version: float = 1.1):
        super().__init__()

        squeezenet = SqueezeNetLoader(squeezenet_version).load(pretrained=True)
        self.backbone = nn.Sequential(*list(squeezenet.children())[0][:12])

        self.fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=6, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 3, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.backbone(x)
        o = self.fc(e)
        o = normalize(torch.sum(torch.sum(o, 2), 2), dim=1)
        return o
