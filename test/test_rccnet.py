import os
import time

import scipy.io as scio
import torch.utils.data

from auxiliary.settings import DEVICE
from auxiliary.utils import print_test_metrics
from classes.data.datasets.TemporalColorConstancy import TemporalColorConstancy
from classes.modules.multiframe.rccnet.ModelRCCNet import ModelRCCNet
from classes.training.Evaluator import Evaluator

"""
Expected results (SqueezeNet): mean: 2.74, med: 2.23, tri: 2.39, bst: 0.75, wst: 5.51, pct: 8.21
"""

BASE_PATH_TO_PRETRAINED = os.path.join("trained_models", "baseline", "rccnet")
PATH_TO_PTH = os.path.join(BASE_PATH_TO_PRETRAINED, "model.pth")


def main():
    evaluator = Evaluator()
    eval_data = {"file_names": [], "predictions": [], "ground_truths": []}

    test_set = TemporalColorConstancy(mode="test")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)
    print('Test set size: {}'.format(len(test_set)))

    model = ModelRCCNet()
    model.evaluation_mode()

    if os.path.exists(PATH_TO_PTH):
        print('Loading pretrained model stored at: {}'.format(PATH_TO_PTH))
        model.load(PATH_TO_PTH)

    model.print_network()

    print("\n --- Testing model RCCNet --- \n")

    with torch.no_grad():
        for i, (seq, mimic, label, file_name) in enumerate(test_loader):
            seq, mimic, label = seq.to(DEVICE), mimic.to(DEVICE), label.to(DEVICE)

            pred = model.predict(seq, mimic)
            loss = model.get_angular_loss(pred, label)

            evaluator.add_error(loss.item())
            eval_data["file_names"].append(file_name[0])
            eval_data["predictions"].append(pred.cpu().numpy())
            eval_data["ground_truths"].append(label.cpu().numpy())

            if i % 10 == 0:
                print("Item {}: {}, AE: {:.4f}".format(i, file_name[0].split(os.sep)[-1], loss.item()))

    eval_data["errors"] = evaluator.get_errors()
    scio.savemat(file_name=os.path.join("results", "rccnet.mat" + str(time.time()) + ".mat"), mdict=eval_data)

    metrics = evaluator.compute_metrics()
    print_test_metrics(metrics)


if __name__ == '__main__':
    main()
