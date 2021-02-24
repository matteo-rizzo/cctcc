import glob
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from auxiliary.settings import DEVICE
from auxiliary.utils import correct, linear_to_nonlinear
from classes.data.datasets.TemporalColorConstancy import TemporalColorConstancy
from classes.modules.multiframe.ctccnet.ModelCTCCNet import ModelCTCCNet
from classes.modules.multiframe.ctccnetc4.ModelCTCCNetC4 import ModelCTCCNetC4

MODEL_TYPE = "ctccnet"
DATA_FOLDER = "tcc_split"

NUM_EXAMPLES = -1
W = -1

PATH_TO_PTH = os.path.join("trained_models", "improved", "best_full_seq", MODEL_TYPE, DATA_FOLDER, "model.pth")
PATH_TO_TEST = os.path.join("dataset", "tcc", "raw", "test")
LOG_DIR = os.path.join("vis", "corrections", "{}_{}_{}".format(MODEL_TYPE, NUM_EXAMPLES, time.time()))

MODELS = {"ctccnet": ModelCTCCNet, "ctccnetc4": ModelCTCCNetC4}


def main():
    log_data = {"file_names": [], "preds1": [], "preds2": [], "preds3": [], "errors": [], "ground_truths": []}

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
        for i, (seq, mimic, label, path_to_data) in enumerate(test_loader):
            if NUM_EXAMPLES != -1 and i >= NUM_EXAMPLES:
                break

            seq, mimic, label = seq.to(DEVICE), mimic.to(DEVICE), label.to(DEVICE)

            o1, o2, o3 = model.predict(seq, mimic)
            pred1, pred2, pred3 = o1, torch.mul(o1, o2), torch.mul(torch.mul(o1, o2), o3)
            loss1 = model.get_angular_loss(pred1, label).item()
            loss2 = model.get_angular_loss(pred2, label).item()
            loss3 = model.get_angular_loss(pred3, label).item()

            log_data["errors"].append((loss1, loss2, loss3))
            log_data["file_names"].append(path_to_data[0])
            log_data["preds1"].append(pred1.cpu().numpy())
            log_data["preds2"].append(pred2.cpu().numpy())
            log_data["preds3"].append(pred3.cpu().numpy())
            log_data["ground_truths"].append(label.cpu().numpy())

            print("Item {}: {} - [ L1: {:.4f} | L2: {:.4f} | L3: {:.4f} ]"
                  .format(i, path_to_data[0].split(os.sep)[-1], loss1, loss2, loss3))

            seq_id = path_to_data[0].split(".")[0].split("test")[1]

            path_to_saved = os.path.join(LOG_DIR, seq_id)
            os.makedirs(path_to_saved)

            paths_to_frames = glob.glob(os.path.join(PATH_TO_TEST, seq_id, "[0-9]*.png"))
            paths_to_frames.sort(key=lambda x: x[:-4].split(os.sep)[-1])

            path_to_corrected_seq_s1 = os.path.join(path_to_saved, "stage_1")
            os.makedirs(path_to_corrected_seq_s1)

            path_to_corrected_seq_s2 = os.path.join(path_to_saved, "stage_2")
            os.makedirs(path_to_corrected_seq_s2)

            path_to_corrected_seq_s3 = os.path.join(path_to_saved, "stage_3")
            os.makedirs(path_to_corrected_seq_s3)

            for paths_to_frame in tqdm(paths_to_frames, desc="Correcting sequence for each stage"):
                original = Image.open(paths_to_frame)
                est1, est2, est3 = correct(original, pred1), correct(original, pred2), correct(original, pred3)
                frame_id = paths_to_frame.split(os.sep)[-1]
                est1.save(os.path.join(path_to_corrected_seq_s1, frame_id))
                est2.save(os.path.join(path_to_corrected_seq_s2, frame_id))
                est3.save(os.path.join(path_to_corrected_seq_s3, frame_id))

            original = Image.open(paths_to_frames[-1])
            gt_corrected = correct(original, label)

            est1, est2, est3 = correct(original, pred1), correct(original, pred2), correct(original, pred3)

            if W > -1:
                h = int(float(original.size[1]) * float(W / float(original.size[0])))
                original = original.resize((W, h), Image.ANTIALIAS)
                gt_corrected = gt_corrected.resize((W, h), Image.ANTIALIAS)
                est1 = est1.resize((W, h), Image.ANTIALIAS)
                est2 = est2.resize((W, h), Image.ANTIALIAS)
                est3 = est3.resize((W, h), Image.ANTIALIAS)

            fig, axs = plt.subplots(1, 5)

            axs[0].imshow(original)
            axs[0].set_title("Original")
            axs[0].axis("off")

            axs[1].imshow(est1)
            axs[1].set_title("Stage 1")
            axs[1].text(0.5, -0.2, "AE: {:.4f}".format(loss1), va="center", ha="center", transform=axs[1].transAxes)
            axs[1].axis("off")

            axs[2].imshow(est2)
            axs[2].set_title("Stage 2")
            axs[2].text(0.5, -0.2, "AE: {:.4f}".format(loss2), va="center", ha="center", transform=axs[2].transAxes)
            axs[2].axis("off")

            axs[3].imshow(est3)
            axs[3].set_title("Stage 3")
            axs[3].text(0.5, -0.2, "AE: {:.4f}".format(loss3), va="center", ha="center", transform=axs[3].transAxes)
            axs[3].axis("off")

            axs[4].imshow(gt_corrected)
            axs[4].set_title("Ground Truth")
            axs[4].axis("off")

            fig.tight_layout(pad=0.5)
            fig.savefig(os.path.join(path_to_saved, "stages.png"), bbox_inches='tight', dpi=200)

            original = linear_to_nonlinear(original)

            original.save(os.path.join(path_to_saved, "original.png"))
            gt_corrected.save(os.path.join(path_to_saved, "gt_corrected.png"))
            est1.save(os.path.join(path_to_saved, "est_corrected_stage_1.png"))
            est2.save(os.path.join(path_to_saved, "est_corrected_stage_2.png"))
            est3.save(os.path.join(path_to_saved, "est_corrected_stage_3.png"))

    pd.DataFrame(log_data).to_csv(os.path.join(LOG_DIR, "evaluation.csv"))


if __name__ == '__main__':
    main()
