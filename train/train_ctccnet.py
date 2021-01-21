from __future__ import print_function

import os
import time

import pandas as pd
import torch.utils.data
from torch.utils.data import DataLoader

from auxiliary.utils import *
from classes.data.datasets.TemporalColorConstancy import TemporalColorConstancy
from classes.modules.multiframe.ctccnet.ModelCTCCNet import ModelCTCCNet
from classes.modules.multiframe.ctccnetc4.ModelCTCCNetC4 import ModelCTCCNetC4
from classes.training.Evaluator import Evaluator
from classes.training.LossTracker import LossTracker

MODEL_TYPE = "ctccnet"
DATA_FOLDER = "fold_0"
BATCH_SIZE = 1
EPOCHS = 2000
LEARNING_RATE = 0.00003
PATH_TO_PTH_SUBMODULE = os.path.join("trained_models", "baseline", "tccnet", "model.pth")

RELOAD_CHECKPOINT = False
PATH_TO_PTH_CHECKPOINT = os.path.join("trained_models", "improved", "best", "fold_0", MODEL_TYPE, "best_model.pth")

MODELS = {"ctccnet": ModelCTCCNet, "ctccnetc4": ModelCTCCNetC4}


def main():
    device = get_device()
    evaluator = Evaluator()

    path_to_log = os.path.join("logs", MODEL_TYPE + "_" + DATA_FOLDER + "_" + str(time.time()))
    os.makedirs(path_to_log, exist_ok=True)
    path_to_metrics = os.path.join(path_to_log, "metrics.csv")

    print("\nLoading data from '{}':".format(DATA_FOLDER))

    training_set = TemporalColorConstancy(mode="train", data_folder=DATA_FOLDER)
    training_set_size = len(training_set)
    print("Training set size: {}".format(training_set_size))
    train_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    test_set = TemporalColorConstancy(mode="test", data_folder=DATA_FOLDER)
    test_set_size = len(test_set)
    print("Test set size: {}\n".format(test_set_size))
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=8)

    model = MODELS[MODEL_TYPE](device)

    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT))
        model.load(PATH_TO_PTH_CHECKPOINT)
    else:
        if PATH_TO_PTH_SUBMODULE != '':
            print('\n Loading pretrained submodules stored at: {} \n'.format(PATH_TO_PTH_SUBMODULE))
            model.load_submodules(PATH_TO_PTH_SUBMODULE)

    model.print_network()
    model.log_network(path_to_log)

    model.set_optimizer(learning_rate=LEARNING_RATE)

    print('\n Training starts... \n')

    best_val_loss = 100.0
    best_mean, best_median, best_trimean = 100.0, 100.0, 100.0
    best_bst25, best_wst25, best_pct95 = 100.0, 100.0, 100.0
    train_l1, train_l2, train_l3, train_mal = LossTracker(), LossTracker(), LossTracker(), LossTracker()
    val_l1, val_l2, val_l3, val_mal = LossTracker(), LossTracker(), LossTracker(), LossTracker()

    for epoch in range(EPOCHS):

        # --- Training ---

        model.train_mode()
        train_l1.reset()
        train_l2.reset()
        train_l3.reset()
        train_mal.reset()
        start = time.time()

        for i, data in enumerate(train_loader):

            model.reset_gradient()

            sequence, mimic, label, file_name = data
            sequence = sequence.unsqueeze(1).to(device) if len(sequence.shape) == 4 else sequence.to(device)
            mimic = mimic.to(device)
            label = label.to(device)

            o1, o2, o3 = model.predict(sequence, mimic)
            l1, l2, l3, mal = model.compute_loss([o1, o2, o3], label)
            mal.backward()
            model.optimize()

            train_l1.update(l1.item())
            train_l2.update(l2.item())
            train_l3.update(l3.item())
            train_mal.update(mal.item())

            if i % 5 == 0:
                print("[ Epoch: {}/{} - Item: {}/{} ] | "
                      "[ Train L1: {:.4f} | Train L2: {:.4f} | Train L3: {:.4f} | Train MAL: {:.4f} ]"
                      .format(epoch, EPOCHS, i, training_set_size, l1.item(), l2.item(), l3.item(), mal.item()))

        train_time = time.time() - start

        # --- Validation ---

        start = time.time()

        val_l1.reset()
        val_l2.reset()
        val_l3.reset()
        val_mal.reset()

        if epoch % 5 == 0:

            print("\n--------------------------------------------------------------")
            print("\t\t Validation")
            print("--------------------------------------------------------------\n")

            with torch.no_grad():

                model.evaluation_mode()
                evaluator.reset_errors()

                for i, data in enumerate(test_loader):

                    sequence, mimic, label, file_name = data
                    sequence = sequence.unsqueeze(1).to(device) if len(sequence.shape) == 4 else sequence.to(device)
                    mimic = mimic.to(device)
                    label = label.to(device)

                    o1, o2, o3 = model.predict(sequence, mimic)
                    l1, l2, l3, mal = model.compute_loss([o1, o2, o3], label)
                    val_l1.update(l1.item())
                    val_l2.update(l2.item())
                    val_l3.update(l3.item())
                    val_mal.update(mal.item())
                    evaluator.add_error(l3.item())

                    if i % 5 == 0:
                        print("[ Epoch: {}/{} - Item: {}/{} ] | "
                              "[ Val L1: {:.4f} | Val L2: {:.4f} | Val L3: {:.4f} | Val MAL: {:.4f} ]"
                              .format(epoch, EPOCHS, i, test_set_size, l1.item(), l2.item(), l3.item(), mal.item()))

            print("\n--------------------------------------------------------------\n")

        val_time = time.time() - start

        metrics = evaluator.compute_metrics()
        print("\n********************************************************************")
        print(" Train Time ... : {:.4f}".format(train_time))
        print(" Train MAL .... : {:.4f}".format(train_mal.avg))
        print(" Train L1 ..... : {:.4f}".format(train_l1.avg))
        print(" Train L2 ..... : {:.4f}".format(train_l2.avg))
        print(" Train L3 ..... : {:.4f}".format(train_l3.avg))
        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ..... : {:.4f}".format(val_time))
            print(" Val MAL ...... : {:.4f}".format(val_mal.avg))
            print(" Val L1 ....... : {:.4f}".format(val_l1.avg))
            print(" Val L2 ....... : {:.4f}".format(val_l2.avg))
            print(" Val L3 ....... : {:.4f} (Best: {:.4f})".format(val_l3.avg, best_val_loss))
            print("....................................................................")
            print(" Mean ......... : {:.4f} (Best: {:.4f})".format(metrics["mean"], best_mean))
            print(" Median ....... : {:.4f} (Best: {:.4f})".format(metrics["median"], best_median))
            print(" Trimean ...... : {:.4f} (Best: {:.4f})".format(metrics["trimean"], best_trimean))
            print(" Best 25% ..... : {:.4f} (Best: {:.4f})".format(metrics["bst25"], best_bst25))
            print(" Worst 25% .... : {:.4f} (Best: {:.4f})".format(metrics["wst25"], best_wst25))
            print(" Worst 5% ..... : {:.4f} (Best: {:.4f})".format(metrics["pct95"], best_pct95))
        print("********************************************************************\n")

        if 0 < val_l3.avg < best_val_loss:
            best_val_loss = val_l3.avg
            best_mean, best_median, best_trimean = metrics["mean"], metrics["median"], metrics["trimean"]
            best_bst25, best_wst25, best_pct95 = metrics["bst25"], metrics["wst25"], metrics["pct95"]
            print("Saving new best model... \n")
            model.save(os.path.join(path_to_log, "model.pth"))

        log_metrics = pd.DataFrame({
            "epoch": [epoch], "lr": [LEARNING_RATE],
            "train_loss": [train_mal.avg], "val_loss": [val_mal.avg], "best_val_loss": [best_val_loss],
            "best_mean": [best_mean], "best_median": best_median, "best_trimean": best_trimean,
            "best_bst25": best_bst25, "best_wst25": best_wst25, "best_pct95": best_pct95,
            **{k: [v] for k, v in metrics.items()}
        })

        log_metrics.to_csv(path_to_metrics,
                           mode='a',
                           header=log_metrics.keys() if not os.path.exists(path_to_metrics) else False,
                           index=False)


if __name__ == '__main__':
    make_deterministic()
    main()
