import os
import time

import torch.utils.data
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE, NUM_STAGES
from auxiliary.utils import log_experiment, print_val_metrics, log_metrics, log_time
from classes.data.datasets.GrayBall import GrayBall
from classes.modules.multiframe.ctccnet2.ModelCTCCNet2 import ModelCTCCNet2
from classes.training.Evaluator import Evaluator
from classes.training.LossTracker import LossTracker

NUM_FOLDS = 3
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.00003
BASE_PATH_TO_PTH_SUBMODULE = os.path.join("trained_models", "gb5", "tccnet")
PATH_TO_LOGS = os.path.join("train", "grayball", "logs")

RELOAD_CHECKPOINT = False
PATH_TO_PTH_CHECKPOINT = os.path.join("trained_models", "ctccnet2", "model.pth")


def main():
    evaluator = Evaluator()

    for n in range(NUM_FOLDS):

        path_to_log = os.path.join(PATH_TO_LOGS, "ctccnet2_fold_{}_{}".format(n, time.time()))
        os.makedirs(path_to_log)

        path_to_metrics_log = os.path.join(path_to_log, "metrics.csv")
        path_to_experiment_log = os.path.join(path_to_log, "experiment.json")

        log_experiment("ctccnet2", "fold_{}".format(n), LEARNING_RATE, path_to_experiment_log)

        print("\n Loading data for FOLD {}:".format(n))

        training_set = GrayBall(mode="train", fold=n, num_folds=NUM_FOLDS, return_labels=True)
        train_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

        test_set = GrayBall(mode="test", fold=n, num_folds=NUM_FOLDS, return_labels=True)
        test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=8)

        training_set_size, test_set_size = len(training_set), len(test_set)
        print("\n TRAINING SET")
        print("\t Size: ..... {}".format(training_set_size))
        print("\t Scenes: ... {}".format(training_set.get_scenes()))
        print("\n TEST SET")
        print("\t Size: ..... {}".format(test_set_size))
        print("\t Scenes: ... {}".format(test_set.get_scenes()))

        model = ModelCTCCNet2()
        model.print_network()
        model.log_network(path_to_log)

        model.set_optimizer(learning_rate=LEARNING_RATE)

        print('\n Training starts... \n')

        best_val_loss, best_metrics = 100.0, evaluator.get_best_metrics()
        train_losses, train_losses_cor, val_losses, val_losses_cor = [], [], [], []
        for _ in range(NUM_STAGES + 1):
            train_losses.append(LossTracker())
            train_losses_cor.append(LossTracker())
            val_losses.append(LossTracker())
            val_losses_cor.append(LossTracker())

        for epoch in range(EPOCHS):

            model.train_mode()
            for tl, tlc in zip(train_losses, train_losses_cor):
                tl.reset()
                tlc.reset()
            start = time.time()

            for i, (seq_temp, seq_shot, labels, _) in enumerate(train_loader):
                seq_temp, seq_shot, labels = seq_temp.to(DEVICE), seq_shot.to(DEVICE), labels.to(DEVICE)
                outputs = model.predict(seq_temp, seq_shot, return_preds=True)
                cas_loss, cas_mal, cor_loss, cor_mal = model.compute_corr_loss(outputs, labels)

                for (tl, sl), (tlc, slc) in zip(zip(train_losses[:-1], cas_loss), zip(train_losses_cor[:-1], cor_loss)):
                    tl.update(sl.item())
                    tlc.update(slc.item())
                train_losses[-1].update(cas_mal.item())
                train_losses[-1].update(cor_mal.item())

                if i % 5 == 0:
                    mal = cas_mal.item() + cor_mal.item()
                    tl_log = " | ".join(["L{}: {:.4f}".format(i + 1, sl.item()) for i, sl in enumerate(cas_loss)])
                    tlc_log = " | ".join(["L{}: {:.4f}".format(i + 1, sl.item()) for i, sl in enumerate(cor_loss)])
                    print(" TRAIN: [ Epoch: {}/{} - Batch: {}/{} ] | Loss: {:.4f} |"
                          " Cascade: [  {} | MAL: {:.4f} ] |"
                          " Corrections: [ {} | MAL: {:.4f} ]"
                          .format(epoch + 1, EPOCHS, i + 1, training_set_size, mal,
                                  tl_log, cas_mal.item(), tlc_log, cor_mal.item()))

            train_time = time.time() - start
            log_time(time=train_time, time_type="train", path_to_log=path_to_experiment_log)

            for vl, vlc in zip(val_losses, val_losses_cor):
                vl.reset()
                vlc.reset()
            start = time.time()

            if epoch % 5 == 0:

                print("\n--------------------------------------------------------------")
                print("\t\t Validation")
                print("--------------------------------------------------------------\n")

                with torch.no_grad():

                    model.evaluation_mode()
                    evaluator.reset_errors()

                    for i, (seq_temp, seq_shot, labels, _) in enumerate(test_loader):
                        seq_temp, seq_shot, labels = seq_temp.to(DEVICE), seq_shot.to(DEVICE), labels.to(DEVICE)
                        outputs = model.predict(seq_temp, seq_shot, return_preds=True)
                        cas_loss, cas_mal, cor_loss, cor_mal = model.get_corr_loss(outputs, labels)

                        losses = zip(zip(val_losses[:-1], cas_loss), zip(val_losses_cor[:-1], cor_loss))
                        for (vl, sl), (vlc, slc) in losses:
                            vl.update(sl.item())
                            vlc.update(slc.item())
                        val_losses[-1].update(cas_mal.item())
                        val_losses[-1].update(cor_mal.item())
                        evaluator.add_error(cas_loss[-1].item())

                        if i % 5 == 0:
                            mal = cas_mal.item() + cor_mal.item()
                            log_cas = ["L{}: {:.4f}".format(i + 1, sl.item()) for i, sl in enumerate(cas_loss)]
                            log_cas = " | ".join(log_cas)
                            log_cor = ["L{}: {:.4f}".format(i + 1, sl.item()) for i, sl in enumerate(cor_loss)]
                            log_cor = " | ".join(log_cor)
                            print(" VAL: [ Epoch: {}/{} - Batch: {}/{} ] | Loss: {:.4f} |"
                                  " Cascade: [  {} | MAL: {:.4f} ] |"
                                  " Corrections: [ {} | MAL: {:.4f} ]"
                                  .format(epoch + 1, EPOCHS, i + 1, test_set_size, mal,
                                          log_cas, cas_mal.item(), log_cor, cor_mal.item()))

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
    main()
