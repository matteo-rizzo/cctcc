from __future__ import print_function

import os
import time

import scipy.io as scio
import torch.utils.data

from auxiliary.utils import get_device, make_deterministic
from classes.data.datasets.TemporalColorConstancy import TemporalColorConstancy
from classes.modules.multiframe.rccnet.ModelRCCNet import ModelRCCNet
from classes.training.Evaluator import Evaluator

"""
Expected results (SqueezeNet): mean: 2.74, med: 2.23, tri: 2.39, bst: 0.75, wst: 5.51, pct: 8.21
"""

BASE_PATH_TO_PRETRAINED = os.path.join("trained_models", "baseline", "rccnet")
PATH_TO_PTH = os.path.join(BASE_PATH_TO_PRETRAINED, "model.pth")


def main():
    device = get_device()
    evaluator = Evaluator()
    log_data = {"file_names": [], "predictions": [], "ground_truths": []}

    test_set = TemporalColorConstancy(mode="test")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)
    print('Test set size: {}'.format(len(test_set)))

    model = ModelRCCNet(device)
    model.evaluation_mode()

    if os.path.exists(PATH_TO_PTH):
        print('Loading pretrained model stored at: {}'.format(PATH_TO_PTH))
        model.load(PATH_TO_PTH)
    else:
        raise ValueError("No pretrained model found at {}".format(PATH_TO_PTH))

    model.print_network()

    print("\n --- Testing model RCCNet --- \n")

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img, mimic, label, file_name = data
            img, mimic, label = img.to(device), mimic.to(device), label.to(device)

            pred = model.predict(img, mimic)
            loss = model.get_angular_loss(pred, label)

            evaluator.add_error(loss.item())
            log_data["file_names"].append(file_name[0])
            log_data["predictions"].append(pred.cpu().numpy())
            log_data["ground_truths"].append(label.cpu().numpy())

            if i % 10 == 0:
                print("Item {}: {}, AE: {:.4f}".format(i, file_name[0].split(os.sep)[-1], loss.item()))

    log_data["errors"] = evaluator.get_errors()
    scio.savemat(file_name=os.path.join("results", "rccnet.mat" + str(time.time()) + ".mat"), mdict=log_data)

    metrics = evaluator.compute_metrics()
    print("\n Mean ............ : {:.4f}".format(metrics["mean"]))
    print(" Median .......... : {:.4f}".format(metrics["median"]))
    print(" Trimean ......... : {:.4f}".format(metrics["trimean"]))
    print(" Best 25% ........ : {:.4f}".format(metrics["bst25"]))
    print(" Worst 25% ....... : {:.4f}".format(metrics["wst25"]))
    print(" Percentile 95 ... : {:.4f} \n".format(metrics["pct95"]))


if __name__ == '__main__':
    make_deterministic()
    main()
