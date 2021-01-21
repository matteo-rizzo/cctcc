from torch.utils import model_zoo

from classes.modules.common.alexnet.AlexNet import AlexNet

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
}


class AlexNetLoader:
    def __init__(self):
        self.__model = AlexNet()

    def load(self, pretrained: bool = False) -> AlexNet:
        """
        @param pretrained: if True, returns a model pre-trained on ImageNet
        """
        if pretrained:
            self.__model.load_state_dict(model_zoo.load_url(model_urls["alexnet"]))
        return self.__model
