import os
import time

import matplotlib.pyplot as plt
import scipy.io as scio
import torch.utils.data

from auxiliary.settings import DEVICE
from classes.data.datasets.TemporalColorConstancy import TemporalColorConstancy
from classes.modules.multiframe.ctccnet.ModelCTCCNet import ModelCTCCNet
from classes.modules.multiframe.ctccnetc4.ModelCTCCNetC4 import ModelCTCCNetC4
from classes.training.Evaluator import Evaluator

"""
Results on the TCC Split:
    * TCCNet    : mean: 1.9944, med: 1.2079, tri: 1.4600, bst: 0.3000, wst: 4.8426, pct: 6.3391
    * CTCCNet   : mean: 1.9505, med: 1.2161, tri: 1.4220, bst: 0.2529, wst: 4.7849, pct: 6.1011
    * CTCCNetC4 : mean: 1.6971, med: 0.9229, tri: 1.1347, bst: 0.2197, wst: 4.3621, pct: 6.0535
"""

MODEL_TYPE = "ctccnet"
DATA_FOLDER = "tcc_split"
PATH_TO_PTH = os.path.join("trained_models", "full_seq", MODEL_TYPE, DATA_FOLDER, "model.pth")

MODELS = {"ctccnet": ModelCTCCNet, "ctccnetc4": ModelCTCCNetC4}


def main():
    eval1, eval2, eval3 = Evaluator(), Evaluator(), Evaluator()
    log_data = {"file_names": [], "predictions": [], "ground_truths": []}

    test_set = TemporalColorConstancy(mode="test", data_folder=DATA_FOLDER)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)
    print('Test set size: {}'.format(len(test_set)))

    model = MODELS[MODEL_TYPE]()

    if os.path.exists(PATH_TO_PTH):
        print('\n Loading pretrained {} model stored at: {} \n'.format(MODEL_TYPE, PATH_TO_PTH))
        model.load(PATH_TO_PTH)
    else:
        raise ValueError("No pretrained {} model found at {}".format(MODEL_TYPE, PATH_TO_PTH))

    model.evaluation_mode()

    print("\n *** Testing model {} on {} *** \n".format(MODEL_TYPE, DATA_FOLDER))

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img, mimic, label, file_name = data
            img, mimic, label = img.to(DEVICE), mimic.to(DEVICE), label.to(DEVICE)

            o1, o2, o3 = model.predict(img, mimic)
            p1, p2, p3 = o1, torch.mul(o1, o2), torch.mul(torch.mul(o1, o2), o3)
            l1 = model.get_angular_loss(p1, label).item()
            l2 = model.get_angular_loss(p2, label).item()
            l3 = model.get_angular_loss(p3, label).item()

            eval1.add_error(l1)
            eval2.add_error(l2)
            eval3.add_error(l3)

            log_data["file_names"].append(file_name[0])
            log_data["predictions"].append(p3.cpu().numpy())
            log_data["ground_truths"].append(label.cpu().numpy())

            if i % 10 == 0:
                print("Item {}: {} - [ L1: {:.4f} | L2: {:.4f} | L3: {:.4f} ]"
                      .format(i, file_name[0].split(os.sep)[-1], l1, l2, l3))

    e1, e2, e3 = eval1.get_errors(), eval2.get_errors(), eval3.get_errors()

    log_data["errors"] = e3
    scio.savemat(file_name=os.path.join("results", MODEL_TYPE + "_" + str(time.time()) + ".mat"), mdict=log_data)

    metrics1, metrics2, metrics3 = eval1.compute_metrics(), eval2.compute_metrics(), eval3.compute_metrics()
    print("\n Mean ............ : [ s1: {:.4f} | s2: {:.4f} | s3: {:.4f} ]"
          .format(metrics1["mean"], metrics2["mean"], metrics3["mean"]))
    print(" Median .......... : [ s1: {:.4f} | s2: {:.4f} | s3: {:.4f} ]"
          .format(metrics1["median"], metrics2["median"], metrics3["median"]))
    print(" Trimean ......... : [ s1: {:.4f} | s2: {:.4f} | s3: {:.4f} ]"
          .format(metrics1["trimean"], metrics2["trimean"], metrics3["trimean"]))
    print(" Best 25% ........ : [ s1: {:.4f} | s2: {:.4f} | s3: {:.4f} ]"
          .format(metrics1["bst25"], metrics2["bst25"], metrics3["bst25"]))
    print(" Worst 25% ....... : [ s1: {:.4f} | s2: {:.4f} | s3: {:.4f} ]"
          .format(metrics1["wst25"], metrics2["wst25"], metrics3["wst25"]))
    print(" Percentile 95 ... : [ s1: {:.4f} | s2: {:.4f} | s3: {:.4f} ] \n"
          .format(metrics1["wst5"], metrics2["wst5"], metrics3["wst5"]))

    plt.plot(range(len(e1)), e1, label="AE1")
    plt.plot(range(len(e2)), e2, label="AE2")
    plt.plot(range(len(e3)), e3, label="AE3")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
