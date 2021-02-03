import re

import numpy as np
import torch

DEVICE_TYPE = "cuda:0"
WARNING = False
RANDOM_SEED = 4


def get_device() -> torch.device:
    """
    Returns the device specified in the experiments parameters (if available, else fallback to a "cpu" device")
    :return: the device specified in the experiments parameters (if available, else fallback to a "cpu" device")
    """

    if DEVICE_TYPE == "cpu":
        return torch.device("cpu")

    if re.match(r"\bcuda:\b\d+", DEVICE_TYPE):
        if not torch.cuda.is_available():
            if WARNING:
                print("WARNING: running on cpu since device {} is not available".format(DEVICE_TYPE))
            return torch.device("cpu")
        return torch.device(DEVICE_TYPE)

    raise ValueError("ERROR: {} is not a valid device! Supported device are 'cpu' and 'cuda:n'".format(DEVICE_TYPE))


def make_deterministic():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
