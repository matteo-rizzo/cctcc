import argparse
import glob
import os
import pickle

import cv2
import numpy as np

USE_HIGH_PRECISION = False
TRUNCATE = False
SUBSEQUENCE_LEN = 2

BASE_PATH_TO_DATA = os.path.join("preprocessed")
BASE_PATH_TO_DATASET = os.path.join("raw", "5f_seqs")


def main(opt):
    truncate = opt.truncate
    subseq_len = opt.subseq_len
    base_path_to_data = opt.path_to_data
    base_path_to_dataset = opt.path_to_dataset

    print("\n=================================================\n")
    print("\t Preprocessing SFU Gray Ball sequences")
    print("\n=================================================\n")

    for scene_name in sorted(os.listdir(base_path_to_dataset)):

        print("\n *** Processing scene {} ***".format(scene_name))

        path_to_data = os.path.join(base_path_to_data, scene_name, "ndata_seq")
        os.makedirs(path_to_data, exist_ok=True)

        path_to_label = os.path.join(base_path_to_data, scene_name, "nlabel")
        os.makedirs(path_to_label, exist_ok=True)

        path_to_labels = os.path.join(base_path_to_data, scene_name, "nlabels")
        os.makedirs(path_to_labels, exist_ok=True)

        path_to_scene = os.path.join(base_path_to_dataset, scene_name)

        for seq_id in sorted(os.listdir(path_to_scene), key=lambda x: int(x)):
            path_to_seq = os.path.join(path_to_scene, seq_id)

            illuminant = list(map(float, open(os.path.join(path_to_seq, "ground_truth.txt"), "r").readlines()))
            illuminants = pickle.load(open(os.path.join(path_to_seq, "seq_ground_truth.pkl"), "rb"))

            files_seq = glob.glob(os.path.join(path_to_seq, "[0-9]*.jpg"))
            files_seq.sort(key=lambda x: x[:-4].split(os.sep)[-1])
            if truncate:
                files_seq = files_seq[len(files_seq) - subseq_len:]

            print("[ Scene: {} | Seq: {} | Len: {}]".format(scene_name, seq_id, len(files_seq)))

            if USE_HIGH_PRECISION:
                images = [np.array(cv2.imread(file, -1), dtype='float32') for file in files_seq]
            else:
                images = [np.array(cv2.imread(file, -1)) for file in files_seq]

            filename = "{}_{}".format(scene_name, seq_id)
            np.save(os.path.join(path_to_data, filename), images)
            np.save(os.path.join(path_to_label, filename), illuminant)
            np.save(os.path.join(path_to_labels, filename), illuminants)

    print("\n=================================================\n")
    print("\t Sequences processed successfully!")
    print("\n=================================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data", type=str, default=BASE_PATH_TO_DATA)
    parser.add_argument('--path_to_dataset', type=str, default=BASE_PATH_TO_DATASET)
    parser.add_argument('--truncate', type=str, default=TRUNCATE)
    parser.add_argument('--subseq_len', type=str, default=SUBSEQUENCE_LEN)
    opt = parser.parse_args()
    main(opt)
