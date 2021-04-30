import argparse
import os
import time

import torch.utils.data
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE, NUM_STAGES
from auxiliary.utils import log_experiment, print_val_metrics, log_metrics, log_time
from classes.data.datasets.TemporalColorConstancy import TemporalColorConstancy
from classes.modules.multiframe.ctccnet2.ModelCTCCNet2 import ModelCTCCNet2
from classes.training.Evaluator import Evaluator
from classes.training.LossTracker import LossTracker

DATA_FOLDER = "tcc_split"
EPOCHS = 2000
LEARNING_RATE = 0.00003

RELOAD_CHECKPOINT = False
PATH_TO_PTH_CHECKPOINT = os.path.join("trained_models", "ctccnet2_{}".format(DATA_FOLDER), "model.pth")


def main(opt):
    data_folder = opt.data_folder
    epochs = opt.epochs
    learning_rate = opt.lr
    evaluator = Evaluator()

    path_to_log = os.path.join("train", "tcc", "logs", "ctccnet2_{}_{}".format(data_folder, str(time.time())))
    os.makedirs(path_to_log)

    path_to_metrics_log = os.path.join(path_to_log, "metrics.csv")
    path_to_experiment_log = os.path.join(path_to_log, "experiment.json")

    log_experiment("ctccnet2", data_folder, learning_rate, path_to_experiment_log)

    print("\n Loading data from '{}':".format(data_folder))

    training_set = TemporalColorConstancy(mode="train", split_folder=data_folder)
    train_loader = DataLoader(dataset=training_set, batch_size=1, shuffle=True, num_workers=8)

    test_set = TemporalColorConstancy(mode="test", split_folder=data_folder)
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=8)

    training_set_size, test_set_size = len(training_set), len(test_set)
    print("Training set size: ... {}".format(training_set_size))
    print("Test set size: ....... {}\n".format(test_set_size))

    model = ModelCTCCNet2()

    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT))
        model.load(PATH_TO_PTH_CHECKPOINT)

    model.print_network()
    model.log_network(path_to_log)

    model.set_optimizer(learning_rate)

    print('\n Training starts... \n')

    best_val_loss, best_metrics = 100.0, evaluator.get_best_metrics()
    train_losses, val_losses = [], []
    for _ in range(NUM_STAGES + 1):
        train_losses.append(LossTracker())
        val_losses.append(LossTracker())

    for epoch in range(epochs):

        model.train_mode()
        for tl in train_losses:
            tl.reset()
        start = time.time()

        for i, (seq_temp, seq_shot, label, _) in enumerate(train_loader):
            seq_temp, seq_shot, label = seq_temp.to(DEVICE), seq_shot.to(DEVICE), label.to(DEVICE)
            outputs = model.predict(seq_temp, seq_shot)
            stages_loss, mal = model.compute_loss(outputs, label)

            for tl, sl in zip(train_losses[:-1], stages_loss):
                tl.update(sl.item())
            train_losses[-1].update(mal.item())

            if i % 5 == 0:
                tl_log = " | ".join(["TL{} {:.4f}".format(i + 1, sl.item()) for i, sl in enumerate(stages_loss)])
                print("[ Epoch: {}/{} - Batch: {}/{} ] | [ {} | Train MAL: {:.4f} ]"
                      .format(epoch + 1, epochs, i + 1, training_set_size, tl_log, stages_loss[-1].item()))

        train_time = time.time() - start
        log_time(time=train_time, time_type="train", path_to_log=path_to_experiment_log)

        for vl in val_losses:
            vl.reset()
        start = time.time()

        if epoch % 5 == 0:

            print("\n--------------------------------------------------------------")
            print("\t\t Validation")
            print("--------------------------------------------------------------\n")

            with torch.no_grad():

                model.evaluation_mode()
                evaluator.reset_errors()

                for i, (seq_temp, seq_shot, label, _) in enumerate(test_loader):
                    seq_temp, seq_shot, label = seq_temp.to(DEVICE), seq_shot.to(DEVICE), label.to(DEVICE)
                    outputs = model.predict(seq_temp, seq_shot)
                    stages_loss, mal = model.get_loss(outputs, label)

                    for vl, sl in zip(val_losses[:-1], stages_loss):
                        vl.update(sl.item())
                    val_losses[-1].update(mal.item())

                    evaluator.add_error(stages_loss[-1].item())

                    if i % 5 == 0:
                        vl_log = ["VL{} {:.4f}".format(i + 1, sl.item()) for i, sl in enumerate(stages_loss)]
                        vl_log = " | ".join(vl_log)
                        print("[ Epoch: {}/{} - Batch: {}/{} ] | [ {} | Val MAL: {:.4f} ]"
                              .format(epoch + 1, epochs, i + 1, test_set_size, vl_log, stages_loss[-1].item()))

            print("\n--------------------------------------------------------------\n")

        val_time = time.time() - start
        log_time(time=val_time, time_type="val", path_to_log=path_to_experiment_log)

        metrics = evaluator.compute_metrics()
        print("\n********************************************************************")
        print(" Train Time ....... : {:.4f}".format(train_time))
        tl_log = " | ".join(["L{} {:.4f}".format(i + 1, tl.avg) for i, tl in enumerate(train_losses[:-1])])
        print(" AVG Train Loss ... : [ {} | MAL: {:.4f} ]".format(tl_log, train_losses[-1].avg))
        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ......... : {:.4f}".format(val_time))
            vl_log = " | ".join(["L{} {:.4f}".format(i + 1, vl.avg) for i, vl in enumerate(val_losses[:-1])])
            print(" AVG Val Loss: .... : [ {} | MAL: {:.4f} ]".format(vl_log, val_losses[-1].avg))
            print("....................................................................")
            print_val_metrics(metrics, best_metrics)
        print("********************************************************************\n")

        curr_val_loss = val_losses[-2].avg
        if 0 < curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            best_metrics = evaluator.update_best_metrics()
            print("Saving new best model... \n")
            model.save(os.path.join(path_to_log, "model.pth"))

        log_metrics(train_losses[-1].avg, val_losses[-1].avg, metrics, best_metrics, path_to_metrics_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default=DATA_FOLDER)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    opt = parser.parse_args()

    print("\n *** Training configuration ***")
    print("\t Data folder ....... : {}".format(opt.data_folder))
    print("\t Epochs ............ : {}".format(opt.epochs))
    print("\t Learning rate ..... : {}".format(opt.lr))

    main(opt)
