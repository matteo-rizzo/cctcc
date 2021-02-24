import os
import time

import torch.utils.data
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from auxiliary.utils import log_experiment, log_metrics, print_metrics, log_time
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

    path_to_log = os.path.join("logs", "tcc", MODEL_TYPE + "_" + DATA_FOLDER + "_" + str(time.time()))
    os.makedirs(path_to_log)

    path_to_metrics_log = os.path.join(path_to_log, "metrics.csv")
    path_to_experiment_log = os.path.join(path_to_log, "experiment.json")

    log_experiment(MODEL_TYPE, DATA_FOLDER, LEARNING_RATE, path_to_experiment_log)

    print("\nLoading data from '{}':".format(DATA_FOLDER))

    training_set = TemporalColorConstancy(mode="train", data_folder=DATA_FOLDER)
    train_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    test_set = TemporalColorConstancy(mode="test", data_folder=DATA_FOLDER)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=8)

    training_set_size, test_set_size = len(training_set), len(test_set)
    print("Training set size: ... {}".format(training_set_size))
    print("Test set size: ....... {}\n".format(test_set_size))

    model = MODELS[MODEL_TYPE]()

    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT))
        model.load(PATH_TO_PTH_CHECKPOINT)

    model.print_network()
    model.log_network(path_to_log)

    model.set_optimizer(learning_rate=LEARNING_RATE)

    print('\n Training starts... \n')

    best_val_loss, best_metrics = 100.0, evaluator.get_best_metrics()
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
                print("[ Epoch: {}/{} - Batch: {}/{} ] | [ Train loss: {:.4f} ]"
                      .format(epoch, EPOCHS, i, training_set_size, loss))

        train_time = time.time() - start
        log_time(time=train_time, time_type="train", path_to_log=path_to_experiment_log)

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
                        print("[ Epoch: {}/{} - Batch: {}/{}] | Val loss: {:.4f} ]"
                              .format(epoch, EPOCHS, i, test_set_size, loss))

            print("\n--------------------------------------------------------------\n")

        val_time = time.time() - start
        log_time(time=val_time, time_type="val", path_to_log=path_to_experiment_log)

        metrics = evaluator.compute_metrics()
        print("\n********************************************************************")
        print(" Train Time ... : {:.4f}".format(train_time))
        print(" Train Loss ... : {:.4f}".format(train_loss.avg))
        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ..... : {:.4f}".format(val_time))
            print(" Val Loss ..... : {:.4f}".format(val_loss.avg))
            print("....................................................................")
            print_metrics(metrics, best_metrics)
        print("********************************************************************\n")

        if 0 < val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            best_metrics = evaluator.update_best_metrics()
            print("Saving new best model... \n")
            model.save(os.path.join(path_to_log, "model.pth"))

        log_metrics(train_loss.avg, val_loss.avg, metrics, best_metrics, path_to_metrics_log)


if __name__ == '__main__':
    main()
