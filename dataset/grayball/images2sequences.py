import glob
import math
import os
import shutil
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import scipy.io
from PIL import Image, ImageDraw

N = 5
ERROR_THRESHOLD = 5.0
MASK_COORDS = [(225, 360), (335, 130)]
PATH_TO_SCENES = "images"

PATH_TO_HIGHLIGHTS = "high_variations"
NUM_HIGHLIGHTS = 10


def mask_ground_truth(path_to_frame: str):
    img = Image.open(path_to_frame).convert('RGB')
    mask = img.copy()
    draw = ImageDraw.Draw(mask)
    draw.rectangle(MASK_COORDS, fill="black")
    return Image.blend(img, mask, 1)


def process_sequence(frame_idx: int, scene_paths: List, path_to_seq: str, images_gt: Dict):
    errors = []
    for i in range(N):
        path_to_frame = scene_paths[frame_idx - i]
        print("\t * Preceding frame {}: {}".format(str(abs(i - N)), path_to_frame))
        # masked_frame = mask_ground_truth(path_to_frame)
        # masked_frame.save(os.path.join(path_to_seq, str(abs(i - N)) + ".jpg"))

        if i < N - 1:
            current_gt = images_gt[path_to_frame]
            preceding_gt = images_gt[scene_paths[frame_idx - i - 1]]
            error = angular_error(current_gt, preceding_gt)
            errors.insert(0, angular_error(current_gt, preceding_gt))
            if error >= ERROR_THRESHOLD:
                print("\n\t -> Detected angle change {:.2f} >= {:.2f} between frames {} and {} \n"
                      .format(error, ERROR_THRESHOLD, frame_idx - i, frame_idx - i - 1))

    mean_error, std_error = np.mean(errors), np.std(errors)

    # plt.plot(range(1, N + 1), errors)
    # plt.title("AVG: {:.4f} - STD DEV: {:.4f}".format(mean_error, std_error))
    # plt.xticks(range(1, N + 1))
    # plt.xlabel("Frame Index")
    # plt.ylabel("Angular Error w.r.t. Preceding Frame")
    # plt.savefig(os.path.join(path_to_seq, "color_trend.png"), bbox_inches='tight', dpi=200)
    # plt.clf()

    return mean_error, std_error


def angular_error(f1: np.ndarray, f2: np.ndarray) -> float:
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

    num_sequences, variations = 0, []

    scenes = os.listdir(PATH_TO_SCENES)
    fold_size = len(scenes) // 3
    test_scenes = [scenes.pop(0 * fold_size) for _ in range(fold_size)]

    for scene_name in os.listdir(PATH_TO_SCENES):
        if scene_name in test_scenes:
            continue

        print("\n *** Processing scene {} ***".format(scene_name))
        scene_paths = sorted(glob.glob(os.path.join(PATH_TO_SCENES, scene_name, "*.jpg")))
        print(scene_paths)

        for frame_idx, path_to_file in enumerate(scene_paths):
            if frame_idx < N - 1:
                continue
            path_to_seq = os.path.join(path_to_sequences, scene_name, str(frame_idx))
            os.makedirs(path_to_seq, exist_ok=True)

            print("\n Processing file {}".format(path_to_file))
            mean_variation, std_variation = process_sequence(frame_idx, scene_paths, path_to_seq, images_gt)
            variations.append((scene_name, frame_idx, path_to_file.split(os.sep)[-1], mean_variation, std_variation))

            # gt = np.array(images_gt[file])
            # np.savetxt(os.path.join(path_to_seq, 'ground_truth.txt'), gt, delimiter=',')

            num_sequences += 1

    print("\n--------------------------------------------------------------------------------------------\n")
    print("\t Generated {} sequences of length N = {} at {}".format(num_sequences, N, path_to_sequences))
    print("\n--------------------------------------------------------------------------------------------\n")

    path_to_save = "{}_{}".format(PATH_TO_HIGHLIGHTS, time.time())
    os.makedirs(path_to_save)

    s, f, fn, mv, sdv = zip(*variations)
    path_to_csv = os.path.join(path_to_save, "data.csv")
    pd.DataFrame({"scene": s, "frame": f, "file_name": fn, "mean_var": mv, "std_dev_var": sdv}).to_csv(path_to_csv)

    print("\n Top {} sequences with largest avg variation \n".format(NUM_HIGHLIGHTS))
    top_avg_seqs = sorted(variations, key=lambda tup: tup[3], reverse=True)[:NUM_HIGHLIGHTS]
    for (s, f, fn, mv, sdv) in top_avg_seqs:
        print(" - Scene: {} | Frame {} (File: {}) | AVG: {:.4f} | STD DEV: {:.4f}".format(s, f, fn, mv, sdv))
        path_to_src = os.path.join(path_to_sequences, s, str(f))
        path_to_dest = os.path.join(path_to_save, "top_avg", s, str(f))
        shutil.copytree(path_to_src, path_to_dest)

    s, f, fn, mv, sdv = zip(*top_avg_seqs)
    path_to_csv = os.path.join(path_to_save, "top_avg.csv")
    pd.DataFrame({"scene": s, "frame": f, "file_name": fn, "mean_var": mv, "std_dev_var": sdv}).to_csv(path_to_csv)

    print("\n Top {} sequences with largest std dev of variations \n".format(NUM_HIGHLIGHTS))
    top_std_dev_seqs = sorted(variations, key=lambda tup: tup[4], reverse=True)[:NUM_HIGHLIGHTS]
    for (s, f, fn, mv, sdv) in top_std_dev_seqs:
        print(" - Scene: {} | Frame {} (File: {}) | AVG: {:.4f} | STD DEV: {:.4f}".format(s, f, fn, mv, sdv))
        path_to_src = os.path.join(path_to_sequences, s, str(f))
        path_to_dest = os.path.join(path_to_save, "top_std_dev", s, str(f))
        shutil.copytree(path_to_src, path_to_dest)

    s, f, fn, mv, sdv = zip(*top_std_dev_seqs)
    path_to_csv = os.path.join(path_to_save, "top_std_dev.csv")
    pd.DataFrame({"scene": s, "frame": f, "file_name": fn, "mean_var": mv, "std_dev_var": sdv}).to_csv(path_to_csv)


if __name__ == '__main__':
    main()
