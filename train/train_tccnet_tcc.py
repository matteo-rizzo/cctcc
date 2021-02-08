import os
import time

import pandas as pd
import torch.utils.data
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from classes.data.datasets.TemporalColorConstancy import TemporalColorConstancy
from classes.modules.multiframe.tccnet.ModelTCCNet import ModelTCCNet
from classes.modules.multiframe.tccnetc4.ModelTCCNetC4 import ModelTCCNetC4
from classes.training.Evaluator import Evaluator
from classes.training.LossTracker import LossTracker

MODEL_TYPE = "tccnetc4"
DATA_FOLDER = "tcc_split"
EPOCHS = 2000
BATCH_SIZE = 1
LEARNING_RATE = 0.00003

RELOAD_CHECKPOINT = False
PATH_TO_PTH_CHECKPOINT = os.path.join("trained_models", "{}_{}".format(MODEL_TYPE, DATA_FOLDER), "model.pth")

MODELS = {"tccnet": ModelTCCNet, "tccnetc4": ModelTCCNetC4}


def main():
    evaluator = Evaluator()

    path_to_log = os.path.join("logs", MODEL_TYPE + "_" + DATA_FOLDER + "_" + str(time.time()))
    os.makedirs(path_to_log, exist_ok=True)
    path_to_metrics = os.path.join(path_to_log, "metrics.csv")

    print("\nLoading data from '{}':".format(DATA_FOLDER))

    training_set = TemporalColorConstancy(mode="train", data_folder=DATA_FOLDER)
    training_set_size = len(training_set)
    print("\t - Training set size: {}".format(training_set_size))
    train_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    test_set = TemporalColorConstancy(mode="test", data_folder=DATA_FOLDER)
    test_set_size = len(test_set)
    print("\t - Test set size: {}\n".format(test_set_size))
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=8)

    model = MODELS[MODEL_TYPE]()

    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT))
        model.load(PATH_TO_PTH_CHECKPOINT)

    model.print_network()
    model.log_network(path_to_log)

    model.set_optimizer(learning_rate=LEARNING_RATE)

    print('\n Training starts... \n')

    best_val_loss = 100.0
    best_mean, best_median, best_trimean = 100.0, 100.0, 100.0
    best_bst25, best_wst25, best_pct95 = 100.0, 100.0, 100.0
    train_loss, val_loss = LossTracker(), LossTracker()

    for epoch in range(EPOCHS):

        # --- Training ---

        model.train_mode()
        train_loss.reset()
        start = time.time()

        for i, data in enumerate(train_loader):

            model.reset_gradient()

            sequence, mimic, label, file_name = data
            sequence = sequence.unsqueeze(1).to(DEVICE) if len(sequence.shape) == 4 else sequence.to(DEVICE)
            mimic = mimic.to(DEVICE)
            label = label.to(DEVICE)

            loss = model.compute_loss(sequence, label, mimic)
            model.optimize()

            train_loss.update(loss)

            if i % 5 == 0:
                print("[ Epoch: {}/{} - Item: {}/{} ] | [ Train loss: {:.4f} ]"
                      .format(epoch, EPOCHS, i, training_set_size, loss))

        train_time = time.time() - start

        # --- Validation ---

        start = time.time()

        val_loss.reset()

        if epoch % 5 == 0:

            print("\n--------------------------------------------------------------")
            print("\t\t Validation")
            print("--------------------------------------------------------------\n")

            with torch.no_grad():

                model.evaluation_mode()
                evaluator.reset_errors()

                for i, data in enumerate(test_loader):

                    sequence, mimic, label, file_name = data
                    sequence = sequence.unsqueeze(1).to(DEVICE) if len(sequence.shape) == 4 else sequence.to(DEVICE)
                    mimic = mimic.to(DEVICE)
                    label = label.to(DEVICE)

                    o = model.predict(sequence, mimic)
                    loss = model.get_angular_loss(o, label).item()
                    val_loss.update(loss)
                    evaluator.add_error(loss)

                    if i % 5 == 0:
                        print("[ Epoch: {}/{} - Item: {}/{}] | Val loss: {:.4f} ]"
                              .format(epoch, EPOCHS, i, test_set_size, loss))

            print("\n--------------------------------------------------------------\n")

        val_time = time.time() - start

        metrics = evaluator.compute_metrics()
        print("\n********************************************************************")
        print(" Train Time ... : {:.4f}".format(train_time))
        print(" Train Loss ... : {:.4f}".format(train_loss.avg))
        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ..... : {:.4f}".format(val_time))
            print(" Val Loss ..... : {:.4f}".format(val_loss.avg))
            print("....................................................................")
            print(" Mean ......... : {:.4f} (Best: {:.4f})".format(metrics["mean"], best_mean))
            print(" Median ....... : {:.4f} (Best: {:.4f})".format(metrics["median"], best_median))
            print(" Trimean ...... : {:.4f} (Best: {:.4f})".format(metrics["trimean"], best_trimean))
            print(" Best 25% ..... : {:.4f} (Best: {:.4f})".format(metrics["bst25"], best_bst25))
            print(" Worst 25% .... : {:.4f} (Best: {:.4f})".format(metrics["wst25"], best_wst25))
            print(" Worst 5% ..... : {:.4f} (Best: {:.4f})".format(metrics["pct95"], best_pct95))
        print("********************************************************************\n")

        if 0 < val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            best_mean, best_median, best_trimean = metrics["mean"], metrics["median"], metrics["trimean"]
            best_bst25, best_wst25, best_pct95 = metrics["bst25"], metrics["wst25"], metrics["pct95"]
            print("Saving new best model... \n")
            model.save(os.path.join(path_to_log, "model.pth"))

        log_metrics = pd.DataFrame({
            "epoch": [epoch], "lr": [LEARNING_RATE],
            "train_loss": [train_loss.avg], "val_loss": [val_loss.avg], "best_val_loss": [best_val_loss],
            "best_mean": [best_mean], "best_median": best_median, "best_trimean": best_trimean,
            "best_bst25": best_bst25, "best_wst25": best_wst25, "best_pct95": best_pct95,
            **{k: [v] for k, v in metrics.items()}
        })

        log_metrics.to_csv(path_to_metrics,
                           mode='a',
                           header=log_metrics.keys() if not os.path.exists(path_to_metrics) else False,
                           index=False)


if __name__ == '__main__':
    main()
