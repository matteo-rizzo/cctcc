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
from classes.modules.multiframe.tccnet.ModelTCCNet import ModelTCCNet
from classes.modules.multiframe.tccnetc4.ModelTCCNetC4 import ModelTCCNetC4
from viz.utils import correct, linear_to_nonlinear

MODEL_TYPE = "tccnetc4"
DATA_FOLDER = "tcc_split"

NUM_EXAMPLES = 1
W = -1

PATH_TO_PTH = os.path.join("trained_models", "improved", "best_full_seq", MODEL_TYPE, DATA_FOLDER, "model.pth")
PATH_TO_TEST = os.path.join("dataset", "tcc", "raw", "test")
LOG_DIR = os.path.join("results", MODEL_TYPE + "_" + str(NUM_EXAMPLES) + "viz_" + str(time.time()))

MODELS = {"tccnet": ModelTCCNet, "tccnetc4": ModelTCCNetC4}


def main():
    log_data = {"file_names": [], "predictions": [], "errors": [], "ground_truths": []}

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

    print("\n *** Generating visualizations for model {} *** \n".format(MODEL_TYPE))

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if NUM_EXAMPLES != -1 and i >= NUM_EXAMPLES:
                break

            seq, mimic, label, path_to_data = data
            seq, mimic, label = seq.to(DEVICE), mimic.to(DEVICE), label.to(DEVICE)

            pred = model.predict(seq, mimic)
            loss = model.get_angular_loss(pred, label).item()

            log_data["errors"].append(loss)
            log_data["file_names"].append(path_to_data[0])
            log_data["predictions"].append(pred.cpu().numpy())
            log_data["ground_truths"].append(label.cpu().numpy())

            print("Item {}: {}, AE: {:.4f}".format(i, path_to_data[0].split(os.sep)[-1], loss))

            seq_id = path_to_data[0].split(".")[0].split("test")[1]

            path_to_saved = os.path.join(LOG_DIR, seq_id)
            os.makedirs(path_to_saved)

            paths_to_frames = glob.glob(os.path.join(PATH_TO_TEST, seq_id, "[0-9]*.png"))
            paths_to_frames.sort(key=lambda x: x[:-4].split(os.sep)[-1])

            original = Image.open(paths_to_frames[-1])
            gt_corrected = correct(original, label)
            est_corrected = correct(original, pred)

            original = linear_to_nonlinear(original)

            if W > -1:
                h = int(float(original.size[1]) * float(W / float(original.size[0])))
                original = original.resize((W, h), Image.ANTIALIAS)
                gt_corrected = gt_corrected.resize((W, h), Image.ANTIALIAS)
                est_corrected = est_corrected.resize((W, h), Image.ANTIALIAS)

            fig, axs = plt.subplots(1, 3)

            axs[0].imshow(original)
            axs[0].set_title("Original")
            axs[0].axis("off")

            axs[1].imshow(est_corrected)
            axs[1].set_title("Prediction")
            axs[1].text(0.5, -0.2, "AE: {:.4f}".format(loss), va="center", ha="center", transform=axs[1].transAxes)
            axs[1].axis("off")

            axs[2].imshow(gt_corrected)
            axs[2].set_title("Ground Truth")
            axs[2].axis("off")

            fig.tight_layout(pad=0.5)
            fig.savefig(os.path.join(path_to_saved, "stages.png"), bbox_inches='tight', dpi=200)

            original.save(os.path.join(path_to_saved, "original.png"))
            est_corrected.save(os.path.join(path_to_saved, "est_corrected.png"))
            gt_corrected.save(os.path.join(path_to_saved, "gt_corrected.png"))

            for path_to_frame in paths_to_frames[:-1]:
                est_corrected = correct(Image.open(path_to_frame), label)
                frame_id = path_to_frame.split(os.sep)[-1]
                est_corrected.save(os.path.join(path_to_saved, "frame_{}_est_corrected.png".format(frame_id)))

    pd.DataFrame(log_data).to_csv(os.path.join(LOG_DIR, "evaluation.csv"))


if __name__ == '__main__':
    main()
