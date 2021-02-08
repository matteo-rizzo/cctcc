import torch
from torch import nn

from auxiliary.settings import DEVICE
from classes.modules.common.alexnet.AlexNetLoader import AlexNetLoader
from classes.modules.common.squeezenet.SqueezeNetLoader import SqueezeNetLoader

"""
RCCNet presented in 'Recurrent Color Constancy' <https://ieeexplore.ieee.org/document/8237844>
Refer to <https://github.com/yanlinqian/Temporal-Color-Constancy> for the original implementation
"""


class RCCNet(nn.Module):

    def __init__(self, input_size: int = 256, hidden_size: int = 128, backbone_type: str = "alexnet"):
        super().__init__()

        self.device = DEVICE
        self.hidden_size = hidden_size

        if backbone_type == "alexnet":
            m1, m2 = AlexNetLoader().load(pretrained=True), AlexNetLoader().load(pretrained=True)
        elif backbone_type == "squeezenet":
            m1, m2 = SqueezeNetLoader().load(pretrained=True), SqueezeNetLoader().load(pretrained=True)
        else:
            raise ValueError("RCC-Net does not support backbone network of type '{}' \n"
                             "Supported backbones are 'alexnet' and 'squeezenet'".format(backbone_type))

        # The names of these nets must be changed, temporarily keeping them for compatibility with trained model
        self.alexnet1_1_A = nn.Sequential(*list(m1.children())[0][:12])
        self.alexnet1_1_B = nn.Sequential(*list(m2.children())[0][:12])

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.lstm_A = nn.LSTMCell(input_size, hidden_size)
        self.lstm_B = nn.LSTMCell(input_size, hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Sigmoid(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def __init_hidden(self, batch_size: int) -> tuple:
        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(self.device)
        cell_state = torch.zeros((batch_size, self.hidden_size)).to(self.device)
        return hidden_state, cell_state

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        @param a: the sequences of frames of shape "bs x ts x nc x h x w"
        @param b: the mimic sequences of shape "bs x ts x nc x h x w"
        @return: the normalized illuminant prediction
        """

        batch_size, time_steps, num_channels, h, w = a.shape
        a = a.view(batch_size * time_steps, num_channels, h, w)
        b = b.view(batch_size * time_steps, num_channels, h, w)
        a = self.alexnet1_1_A(a)
        b = self.alexnet1_1_B(b)

        a = torch.mean(a, dim=(2, 3))
        b = torch.mean(b, dim=(2, 3))
        a = a.view(batch_size, time_steps, -1)
        b = b.view(batch_size, time_steps, -1)

        hidden_state_1, cell_state_1 = self.__init_hidden(batch_size)
        hidden_state_2, cell_state_2 = self.__init_hidden(batch_size)

        for t in range(a.shape[1]):
            hidden_state_1, cell_state_1 = self.lstm_A(a[:, t, :], (hidden_state_1, cell_state_1))
            hidden_state_2, cell_state_2 = self.lstm_B(b[:, t, :], (hidden_state_2, cell_state_2))

        c = torch.cat((hidden_state_1, hidden_state_2), 1)
        c = self.fc(c)

        return torch.nn.functional.normalize(c, dim=1)
