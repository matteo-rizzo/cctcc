import torch
from torch import nn
from torch.nn.functional import normalize

from auxiliary.settings import DEVICE
from classes.modules.common.conv_lstm.ConvLSTMCell import ConvLSTMCell
from classes.modules.multiframe.tccnetc4.submodules.C4 import C4

""" TemporalColorConstancy-Net presented in https://arxiv.org/abs/2003.03763, 'A Benchmark for Temporal Color Constancy' """


class TCCNetC4(nn.Module):

    def __init__(self, hidden_size: int = 128, kernel_size: int = 5):
        super().__init__()

        self.device = DEVICE
        self.hidden_size = hidden_size

        self.c4_A, self.c4_B = C4(), C4()

        self.lstm_A = ConvLSTMCell(512, self.hidden_size, kernel_size)
        self.lstm_B = ConvLSTMCell(512, self.hidden_size, kernel_size)

        # Hidden size is halved with respect to training
        self.fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(self.hidden_size * 2, self.hidden_size // 2, kernel_size=6, stride=1, padding=3),
            nn.Sigmoid(),
            nn.Conv2d(self.hidden_size // 2, 3, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def __init_hidden(self, batch_size: int, h: int, w: int) -> tuple:
        hidden_state = torch.zeros((batch_size, self.hidden_size, h, w)).to(self.device)
        cell_state = torch.zeros((batch_size, self.hidden_size, h, w)).to(self.device)
        return hidden_state, cell_state

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        @param a: the sequences of frames of shape "bs x ts x nc x h x w"
        @param b: the mimic sequences of shape "bs x ts x nc x h x w"
        @return: the normalized illuminant prediction
        """

        batch_size, time_steps, num_channels, h, w = a.shape

        a = a.view(batch_size * time_steps, num_channels, h, w)
        a = self.c4_A(a)
        _, num_channels_a, h_a, w_a = a.shape
        a = a.view(batch_size, time_steps, num_channels_a, h_a, w_a)

        b = b.view(batch_size * time_steps, num_channels, h, w)
        b = self.c4_B(b)
        _, num_channels_b, h_b, w_b = b.shape
        b = b.view(batch_size, time_steps, num_channels_b, h_b, w_b)

        self.lstm_A.init_hidden(self.hidden_size, (h_a, w_a))
        hidden_state_1, cell_state_1 = self.__init_hidden(batch_size, h_a, w_a)

        self.lstm_B.init_hidden(self.hidden_size, (h_b, w_b))
        hidden_state_2, cell_state_2 = self.__init_hidden(batch_size, h_b, w_b)

        for t in range(a.shape[1]):
            hidden_state_1, cell_state_1 = self.lstm_A(a[:, t, :], hidden_state_1, cell_state_1)
            hidden_state_2, cell_state_2 = self.lstm_B(b[:, t, :], hidden_state_2, cell_state_2)

        c = torch.cat((hidden_state_1, hidden_state_2), 1)
        c = self.fc(c)

        return normalize(c if len(c.shape) == 2 else torch.sum(torch.sum(c, 2), 2), dim=1)
