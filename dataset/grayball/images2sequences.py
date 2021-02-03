import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from PIL import Image, ImageDraw

N = 5
ERROR_THRESHOLD = 3.0
PATH_TO_SCENES = "images"
MASK_COORDS = [(225, 360), (335, 130)]


def mask_ground_truth(path_to_frame: str):
    img = Image.open(path_to_frame).convert('RGB')

    mask = img.copy()
    draw = ImageDraw.Draw(mask)
    draw.rectangle(MASK_COORDS, fill="black")

    return Image.blend(img, mask, 1)


def process_sequence(frame_idx: int, scene_paths: list, path_to_seq: str, images_gt: dict):
    errors = []
    for i in range(N):
        path_to_frame = scene_paths[frame_idx - i]
        print("\t * Preceding frame {}: {}".format(str(abs(i - N)), path_to_frame))
        masked_frame = mask_ground_truth(path_to_frame)
        masked_frame.save(os.path.join(path_to_seq, str(abs(i - N)) + ".jpg"))

        if i < N - 1:
            current_gt, preceding_gt = images_gt[path_to_frame], images_gt[scene_paths[frame_idx - i - 1]]
            error = angular_error(current_gt, preceding_gt)
            errors.insert(0, angular_error(current_gt, preceding_gt))
            if error >= ERROR_THRESHOLD:
                print("\n\t -> Detected angle change {:.2f} >= {:.2f} between frames {} and {} \n"
                      .format(error, ERROR_THRESHOLD, frame_idx - i, frame_idx - i - 1))

    plt.plot(range(1, len(errors) + 1), errors)
    plt.title("AVG: {:.4f} - STD DEV: {:.4f}".format(np.mean(errors), np.std(errors)))
    plt.xticks(range(1, len(errors) + 1))
    plt.xlabel("Frame Index")
    plt.ylabel("Angular Error w.r.t. Preceding Frame")
    plt.savefig(os.path.join(path_to_seq, "color_trend.png"), bbox_inches='tight', dpi=200)
    plt.clf()


def angular_error(f1: np.array, f2: np.array) -> float:
    return np.arccos(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))) * (180 / math.pi)


def main():
    images = np.array([x.strip() for x in open("images_gt_order.lst").readlines()])
    ground_truths = scipy.io.loadmat("groundtruth.mat")["real_rgb"].tolist()
    pd.DataFrame({"image": images, "ground_truth": ground_truths}).to_csv("images_gt.csv", index=False)

    images_gt = {img: gt for img, gt in zip(images, ground_truths)}

    path_to_sequences = os.path.join("raw", "{}f_seqs".format(N))
    os.makedirs(path_to_sequences, exist_ok=True)

    print("\n--------------------------------------------------------------------------------------------\n")
    print("\tPreprocessing SFU Gray Ball for N = {}".format(N))
    print("\n--------------------------------------------------------------------------------------------\n")

    num_sequences = 0

    for scene_name in os.listdir(PATH_TO_SCENES):

        print("\n *** Processing scene {} ***".format(scene_name))
        scene_paths = sorted(glob.glob(os.path.join(PATH_TO_SCENES, scene_name, "*.jpg")))

        for frame_idx, file in enumerate(scene_paths):
            if frame_idx < N - 1:
                continue
            path_to_seq = os.path.join(path_to_sequences, scene_name, str(frame_idx))
            os.makedirs(path_to_seq, exist_ok=True)

            print("\n Processing file {}".format(file))
            process_sequence(frame_idx, scene_paths, path_to_seq, images_gt)

            gt = np.array(images_gt[file])
            np.savetxt(os.path.join(path_to_seq, 'ground_truth.txt'), gt, delimiter=',')

            num_sequences += 1

    print("\n--------------------------------------------------------------------------------------------\n")
    print("\t Generated {} sequences of length N = {} at {}".format(num_sequences, N, path_to_sequences))
    print("\n--------------------------------------------------------------------------------------------\n")


if __name__ == '__main__':
    main()
