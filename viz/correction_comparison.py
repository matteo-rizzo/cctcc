import glob
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from classes.data.datasets.TemporalColorConstancy import TemporalColorConstancy
from classes.modules.multiframe.ctccnet.ModelCTCCNet import ModelCTCCNet
from classes.modules.multiframe.ctccnetc4.ModelCTCCNetC4 import ModelCTCCNetC4
from classes.modules.multiframe.tccnet.ModelTCCNet import ModelTCCNet
from classes.modules.multiframe.tccnetc4.ModelTCCNetC4 import ModelTCCNetC4
from viz.utils import correct, linear_to_nonlinear

DATA_FOLDER = "tcc_split"

NUM_EXAMPLES = -1
W = -1

BASE_PATH_TO_PTH = os.path.join("trained_models", "improved", "best_full_seq")
DATA_TYPE = "test"
PATH_TO_DATASET = os.path.join("dataset", "tcc", "raw", DATA_TYPE)
LOG_DIR = os.path.join("results", "models_comparison_" + str(NUM_EXAMPLES) + "_" + DATA_TYPE + "_" + str(time.time()))

MODELS = {"tccnet": ModelTCCNet, "tccnetc4": ModelTCCNetC4, "ctccnet": ModelCTCCNet, "ctccnetc4": ModelCTCCNetC4}


def main():
    log_data = {"file_names": [], "ground_truths": []}
    models = {}

    test_set = TemporalColorConstancy(mode=DATA_TYPE, data_folder=DATA_FOLDER)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)
    print('Test set size: {}'.format(len(test_set)))

    for model_type, model in MODELS.items():
        model = model()

        path_to_pth = os.path.join(BASE_PATH_TO_PTH, model_type, DATA_FOLDER, "model.pth")
        if os.path.exists(path_to_pth):
            print('\n Loading pretrained {} model stored at: {} \n'.format(model_type, path_to_pth))
            model.load(path_to_pth)
        else:
            raise ValueError("No pretrained {} model found at {}".format(model_type, path_to_pth))

        model.evaluation_mode()
        models[model_type] = model
        log_data[model_type + "_preds"], log_data[model_type + "_errors"] = [], []

    print("\n *** Generating comparison of visualizations *** \n")

    with torch.no_grad():
        for i, data in enumerate(test_loader):

            if NUM_EXAMPLES != -1 and i >= NUM_EXAMPLES:
                break

            img, mimic, label, path_to_data = data
            img, mimic, label = img.to(DEVICE), mimic.to(DEVICE), label.to(DEVICE)

            log_data["file_names"].append(path_to_data[0])
            log_data["ground_truths"].append(label.cpu().numpy())

            seq_id = path_to_data[0].split(".")[0].split(DATA_TYPE)[1]

            path_to_saved = os.path.join(LOG_DIR, seq_id)
            os.makedirs(path_to_saved)

            paths_to_frames = glob.glob(os.path.join(PATH_TO_DATASET, seq_id, "[0-9]*.png"))
            paths_to_frames.sort(key=lambda x: x[:-4].split(os.sep)[-1])

            original = Image.open(paths_to_frames[-1])
            gt_corrected = correct(original, label)

            if W > -1:
                h = int(float(original.size[1]) * float(W / float(original.size[0])))
                original = original.resize((W, h), Image.ANTIALIAS)
                gt_corrected = gt_corrected.resize((W, h), Image.ANTIALIAS)

            fig, axs = plt.subplots(1, 2 + len(models.keys()))

            axs[0].imshow(linear_to_nonlinear(original))
            axs[0].set_title("Original")
            axs[0].axis("off")

            axs[1].imshow(gt_corrected)
            axs[1].set_title("Ground Truth")
            axs[1].axis("off")

            losses = {}

            for j, (model_type, model) in enumerate(models.items()):
                pred = model.predict(img, mimic)
                if isinstance(pred, tuple):
                    pred = torch.mul(torch.mul(pred[0], pred[1]), pred[2])
                loss = model.get_angular_loss(pred, label).item()

                losses[model_type] = loss
                log_data[model_type + "_preds"].append(pred.cpu().numpy())
                log_data[model_type + "_errors"].append(loss)

                color_fig, color_axs = plt.subplots(1, 3)

                color_axs[0].add_patch(plt.Circle((0, 0), radius=1, fc=tuple(pred.cpu().numpy()[0])))
                color_axs[0].set_title("Predicted")
                color_axs[0].axis("scaled")
                color_axs[0].axis("off")

                color_axs[1].add_patch(plt.Circle((0, 0), radius=1, fc=tuple(label.cpu().numpy()[0])))
                color_axs[1].set_title("Ground Truth")
                color_axs[1].axis("scaled")
                color_axs[1].axis("off")

                color_axs[2].add_patch(plt.Circle((0, 0), radius=1, fc=tuple(pred.cpu().numpy()[0])))
                color_axs[2].add_patch(plt.Circle((0, 0), radius=0.6, fc=tuple(label.cpu().numpy()[0])))
                color_axs[2].set_title("Comparison")
                color_axs[2].axis("scaled")
                color_axs[2].axis("off")

                color_fig.tight_layout()
                color_fig.savefig(os.path.join(path_to_saved, model_type + "_pred_gt_comparison.png"),
                                  bbox_inches='tight', dpi=200)
                plt.close(color_fig)

                est_corrected = correct(original, pred)

                if W > -1:
                    h = int(float(original.size[1]) * float(W / float(original.size[0])))
                    est_corrected = est_corrected.resize((W, h), Image.ANTIALIAS)

                est_corrected.save(os.path.join(path_to_saved, model_type + "_corrected.png"))

                axs[2 + j].imshow(est_corrected)
                axs[2 + j].set_title(model_type)
                axs[2 + j].axis("off")
                axs[2 + j].text(0.5, -0.2, "AE: {:.4f}".format(loss),
                                va="center", ha="center", transform=axs[2 + j].transAxes)

            print("Item {}: {} - AE: [ {} ]".format(i, path_to_data[0].split(os.sep)[-1],
                                                    " | ".join(["{}: {:.4f}".format(m, l) for m, l in losses.items()])))

            # fig.tight_layout(pad=0.5)
            fig.savefig(os.path.join(path_to_saved, "comparison.png"), bbox_inches='tight', dpi=200)
            plt.close(fig)

            linear_to_nonlinear(original).save(os.path.join(path_to_saved, "original.png"))
            gt_corrected.save(os.path.join(path_to_saved, "gt_corrected.png"))

    pd.DataFrame(log_data).to_csv(os.path.join(LOG_DIR, "evaluation.csv"))


if __name__ == '__main__':
    main()
