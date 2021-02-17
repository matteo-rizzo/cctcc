import os
import time

import scipy.io as scio
import torch.utils.data
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from classes.data.datasets.TemporalColorConstancy import TemporalColorConstancy
from classes.modules.multiframe.tccnet.ModelTCCNet import ModelTCCNet
from classes.modules.multiframe.tccnetc4.ModelTCCNetC4 import ModelTCCNetC4
from classes.training.Evaluator import Evaluator

"""
Results on the TCC Split:
    * TCCNet   : mean: 1.9944, med: 1.2079, tri: 1.4600, bst: 0.3000, wst: 4.8426, pct: 6.3391
    * TCCNetC4 : mean: 1.7171, med: 1.0847, tri: 1.2002, bst: 0.1982, wst: 4.3331, pct: 6.0090
"""

MODEL_TYPE = "tccnetc4"
DATA_FOLDER = "tcc_split"
PATH_TO_PTH = os.path.join("trained_models", "full_seq", MODEL_TYPE, DATA_FOLDER, "model.pth")

MODELS = {"tccnet": ModelTCCNet, "tccnetc4": ModelTCCNetC4}


def main():
    evaluator = Evaluator()
    log_data = {"file_names": [], "predictions": [], "ground_truths": []}

    test_set = TemporalColorConstancy(mode="test", data_folder=DATA_FOLDER)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)
    print('Test set size: {}'.format(len(test_set)))

    model = MODELS[MODEL_TYPE]()

    if os.path.exists(PATH_TO_PTH):
        print('\n Loading pretrained {} model stored at: {} \n'.format(MODEL_TYPE, PATH_TO_PTH))
        model.load(PATH_TO_PTH)
    else:
        raise ValueError("No pretrained {} model found at {}".format(MODEL_TYPE, PATH_TO_PTH))

    model.evaluation_mode()
    model.print_network()

    print("\n *** Testing model {} on {} *** \n".format(MODEL_TYPE, DATA_FOLDER))

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            seq, mimic, label, file_name = data
            seq, mimic, label = seq.to(DEVICE), mimic.to(DEVICE), label.to(DEVICE)

            pred = model.predict(seq, mimic)
            loss = model.get_angular_loss(pred, label).item()

            evaluator.add_error(loss)
            log_data["file_names"].append(file_name[0])
            log_data["predictions"].append(pred.cpu().numpy())
            log_data["ground_truths"].append(label.cpu().numpy())

            if i % 10 == 0:
                print("Item {}: {}, AE: {:.4f}".format(i, file_name[0].split(os.sep)[-1], loss))

    log_data["errors"] = evaluator.get_errors()
    scio.savemat(file_name=os.path.join("results", MODEL_TYPE + "_" + str(time.time()) + ".mat"), mdict=log_data)

    metrics = evaluator.compute_metrics()
    print("\n Mean ............ : {:.4f}".format(metrics["mean"]))
    print(" Median .......... : {:.4f}".format(metrics["median"]))
    print(" Trimean ......... : {:.4f}".format(metrics["trimean"]))
    print(" Best 25% ........ : {:.4f}".format(metrics["bst25"]))
    print(" Worst 25% ....... : {:.4f}".format(metrics["wst25"]))
    print(" Percentile 95 ... : {:.4f} \n".format(metrics["wst5"]))


if __name__ == '__main__':
    main()
