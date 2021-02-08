import torch
from torch.autograd import Variable

from classes.modules.common.BaseModel import BaseModel
from classes.modules.singleframe.fc4.FC4 import FC4


class ModelFC4(BaseModel):

    def __init__(self, DEVICE: torch.DEVICE):
        super().__init__(DEVICE)
        self._network = FC4().to(self._device)

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        return self._network(img)

    def compute_loss(self, img: torch.Tensor, label: torch.Tensor) -> float:
        pred = self.predict(img)
        loss = Variable(self.get_angular_loss(pred, label), requires_grad=True)
        loss.backward()
        return loss.item()
