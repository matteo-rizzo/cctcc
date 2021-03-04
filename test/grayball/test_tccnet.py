import argparse
import os
from time import time, perf_counter

import numpy as np
import pandas as pd
import torch.utils.data
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from auxiliary.utils import print_test_metrics
from classes.data.datasets.GrayBall import GrayBall
from classes.modules.multiframe.tccnet.ModelTCCNet import ModelTCCNet
from classes.modules.multiframe.tccnetc4.ModelTCCNetC4 import ModelTCCNetC4
from classes.training.Evaluator import Evaluator

"""
Results on the TCC Split:
    * TCCNet   : mean: 1.9944, med: 1.2079, tri: 1.4600, bst: 0.3000, wst: 4.8426, pct: 6.3391
    * TCCNetC4 : mean: 1.7171, med: 1.0847, tri: 1.2002, bst: 0.1982, wst: 4.3331, pct: 6.0090
"""

MODEL_TYPE = "tccnet"
DATA_FOLDER = "gb5"
SPLIT_FOLDER = "fold_0"
PATH_TO_LOGS = os.path.join("test", "grayball", "logs")

MODELS = {"tccnet": ModelTCCNet, "tccnetc4": ModelTCCNetC4}


def main(opt):
    model_type = opt.model_type
    data_folder = opt.data_folder
    split_folder = opt.split_folder

    path_to_pth = os.path.join("trained_models", data_folder, model_type, split_folder, "model.pth")
    path_to_log = os.path.join(PATH_TO_LOGS, "{}_{}_{}_{}".format(model_type, data_folder, split_folder, time()))
    os.makedirs(path_to_log)

    evaluator = Evaluator()
    eval_data = {"file_names": [], "predictions": [], "ground_truths": []}
    inference_times = []

    test_set = GrayBall(mode="test", num_folds=1)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)
    print('Test set size: {}'.format(len(test_set)))

    model = MODELS[model_type]()

    print('\n Loading pretrained {} model stored at: {} \n'.format(model_type, path_to_pth))
    model.load(path_to_pth)
    model.evaluation_mode()

    print("\n *** Testing model {} on {}/{} *** \n".format(model_type, data_folder, split_folder))

    with torch.no_grad():
        for i, (seq, mimic, label, file_name) in enumerate(test_loader):
            seq, mimic, label = seq.to(DEVICE), mimic.to(DEVICE), label.to(DEVICE)

            tic = perf_counter()
            pred = model.predict(seq, mimic)
            toc = perf_counter()
            inference_times.append(toc - tic)

            loss = model.get_angular_loss(pred, label).item()

            evaluator.add_error(loss)
            eval_data["file_names"].append(file_name[0])
            eval_data["predictions"].append(pred.cpu().numpy())
            eval_data["ground_truths"].append(label.cpu().numpy())

            if i % 1 == 0:
                print(" - Item {}: {}, AE: {:.4f}".format(i, file_name[0].split(os.sep)[-1], loss))

    print(" \n Average inference time: {:.4f} \n".format(np.mean(inference_times)))

    eval_data["errors"] = evaluator.get_errors()
    metrics = evaluator.compute_metrics()
    print_test_metrics(metrics)

    pd.DataFrame({k: [v] for k, v in metrics.items()}).to_csv(os.path.join(path_to_log, "metrics.csv"), index=False)
    pd.DataFrame(eval_data).to_csv(os.path.join(path_to_log, "eval.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE)
    parser.add_argument('--data_folder', type=str, default=DATA_FOLDER)
    parser.add_argument('--split_folder', type=str, default=SPLIT_FOLDER)
    opt = parser.parse_args()
    main(opt)
