import torch
import torch.nn as nn

from classes.modules.common.conv_lstm.ConvLSTMCell import ConvLSTMCell

"""
A multi-layer convolutional LSTM module based on: https://arxiv.org/abs/1506.04214,
"Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting" 
"""


class ConvLSTM(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_steps=(1,)):
        """
        @param input_channels: the first input feature map
        @param hidden_channels: a list of succeeding lstm layers
        @param kernel_size:
        @param step:
        @param effective_steps:
        """
        super().__init__()

        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_steps = effective_steps
        self._all_layers = []

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, inputs: torch.Tensor) -> tuple:
        internal_state, outputs = [], []
        x, new_c = None, None
        for step in range(self.step):
            x = inputs
            for i in range(self.num_layers):

                # All cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(self.hidden_channels[i], (height, width))
                    internal_state.append((h, c))

                # Execute forward
                (h, c) = internal_state[i]
                print('h,c', h.shape, c.shape)
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)

            # Record effective steps only
            if step in self.effective_steps:
                outputs.append(x)

        return outputs, (x, new_c)
