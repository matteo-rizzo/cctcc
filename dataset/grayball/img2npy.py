import glob
import os

import cv2
import numpy as np

USE_HIGH_PRECISION = False
TRUNCATE = False
SUBSEQUENCE_LEN = 2

BASE_PATH_TO_DATA = os.path.join("prep")
BASE_PATH_TO_DATASET = os.path.join("raw", "5f_seqs")
GROUND_TRUTH_FILE = "ground_truth.txt"


def main():
    print("\n=================================================\n")
    print("\t Preprocessing SFU Gray Ball sequences")
    print("\n=================================================\n")

    for scene_name in os.listdir(BASE_PATH_TO_DATASET):

        print("\n *** Processing scene {} ***".format(scene_name))

        path_to_data = os.path.join(BASE_PATH_TO_DATA, scene_name, "ndata_seq")
        os.makedirs(path_to_data, exist_ok=True)

        path_to_label = os.path.join(BASE_PATH_TO_DATA, scene_name, "nlabel")
        os.makedirs(path_to_label, exist_ok=True)

        path_to_scene = os.path.join(BASE_PATH_TO_DATASET, scene_name)

        for seq_id in os.listdir(path_to_scene):
            path_to_seq = os.path.join(path_to_scene, seq_id)

            illuminant = list(map(float, open(os.path.join(path_to_seq, GROUND_TRUTH_FILE), "r").readlines()))

            files_seq = glob.glob(os.path.join(path_to_seq, "[0-9]*.jpg"))
            files_seq.sort(key=lambda x: x[:-4].split(os.sep)[-1])
            if TRUNCATE:
                files_seq = files_seq[len(files_seq) - SUBSEQUENCE_LEN:]

            print("[ Scene: {} | Seq: {} | Len: {}]".format(scene_name, seq_id, len(files_seq)))

            if USE_HIGH_PRECISION:
                images = [np.array(cv2.imread(file, -1), dtype='float32') for file in files_seq]
            else:
                images = [np.array(cv2.imread(file, -1)) for file in files_seq]

            filename = "{}_{}".format(scene_name, seq_id)
            np.save(os.path.join(path_to_data, filename), images)
            np.save(os.path.join(path_to_label, filename), illuminant)


print("\n=================================================\n")
print("\t Sequences processed successfully!")
print("\n=================================================\n")

if __name__ == '__main__':
    main()
