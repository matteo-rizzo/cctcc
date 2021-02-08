import json
import os

import pandas as pd


def log_experiment(model_type: str, data_folder: str, lr: float, path_to_log: str):
    experiment_data = {
        "model_type": model_type,
        "data_folder": data_folder,
        "learning_rate": lr,
        "train_time": 0,
        "val_time": 0
    }
    json.dump(experiment_data, open(path_to_log, 'w'), indent=2)


def log_time(time: float, time_type: str, path_to_log: str):
    data = json.load(open(path_to_log, 'r'))
    data["{}_time".format(time_type)] += time
    open(path_to_log, 'w+').write(json.dumps(data, indent=2))


def log_metrics(train_loss: float, val_loss: float, current_metrics: dict, best_metrics: dict, path_to_log: str):
    log_data = pd.DataFrame({
        "train_loss": [train_loss],
        "val_loss": [val_loss],
        "best_mean": best_metrics["mean"],
        "best_median": best_metrics["median"],
        "best_trimean": best_metrics["trimean"],
        "best_bst25": best_metrics["bst25"],
        "best_wst25": best_metrics["wst25"],
        "best_wst5": best_metrics["wst5"],
        **{k: [v] for k, v in current_metrics.items()}
    })
    log_data.to_csv(path_to_log,
                    mode='a',
                    header=log_data.keys() if not os.path.exists(path_to_log) else False,
                    index=False)


def print_metrics(current_metrics: dict, best_metrics: dict):
    print(" Mean ......... : {:.4f} (Best: {:.4f})".format(current_metrics["mean"], best_metrics["mean"]))
    print(" Median ....... : {:.4f} (Best: {:.4f})".format(current_metrics["median"], best_metrics["median"]))
    print(" Trimean ...... : {:.4f} (Best: {:.4f})".format(current_metrics["trimean"], best_metrics["trimean"]))
    print(" Best 25% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["bst25"], best_metrics["bst25"]))
    print(" Worst 25% .... : {:.4f} (Best: {:.4f})".format(current_metrics["wst25"], best_metrics["wst25"]))
    print(" Worst 5% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["wst5"], best_metrics["wst5"]))
