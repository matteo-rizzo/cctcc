import glob
import os

import cv2
import numpy as np
import pandas as pd

USE_CV_METADATA = False
USE_VAL_SET = False
FOLD_NUM = 2
CV_METADATA_FILE = "3_folds_experiment.csv"

USE_HIGH_PRECISION = False
DOWN_SAMPLE = False
TRUNCATE = True
SUBSEQUENCE_LEN = 2

BASE_PATH_TO_DATA = os.path.join("preprocessed", "fold_" + str(FOLD_NUM) if USE_CV_METADATA else "tcc_split")
if DOWN_SAMPLE:
    BASE_PATH_TO_DATA += "_ds"
if TRUNCATE:
    BASE_PATH_TO_DATA += "_{}f".format(SUBSEQUENCE_LEN)

PATH_TO_NUMPY_SEQ = os.path.join(BASE_PATH_TO_DATA, "ndata_seq")
PATH_TO_NUMPY_LABEL = os.path.join(BASE_PATH_TO_DATA, "nlabel")

BASE_PATH_TO_DATASET = "raw"
PATH_TO_TRAIN = os.path.join(BASE_PATH_TO_DATASET, "train")
PATH_TO_TEST = os.path.join(BASE_PATH_TO_DATASET, "test")
GROUND_TRUTH_FILE = "groundtruth.txt"


def main():
    print("\n=================================================\n")
    print("\t Preprocessing TCC sequences")
    print("\n=================================================\n")

    os.makedirs(PATH_TO_NUMPY_SEQ, exist_ok=True)
    os.makedirs(PATH_TO_NUMPY_LABEL, exist_ok=True)

    if USE_CV_METADATA:

        if USE_VAL_SET:
            metadata = pd.read_csv(CV_METADATA_FILE, converters={'train': eval, 'val': eval, 'test': eval})
        else:
            metadata = pd.read_csv(CV_METADATA_FILE, converters={'train': eval, 'test': eval})

        for set_type in metadata.columns:
            convert_data(metadata.iloc[FOLD_NUM][set_type], set_type)
    else:
        convert_data(file_names=glob.glob(os.path.join(PATH_TO_TRAIN, "*")), set_type="train")
        convert_data(file_names=glob.glob(os.path.join(PATH_TO_TEST, "*")), set_type="test")

    print("\n=================================================\n")
    print("\t Sequences processed successfully!")
    print("\n=================================================\n")


def save_sequence(files_seq: list, illuminant: list, filename: str):
    if USE_HIGH_PRECISION:
        images = [np.array(cv2.imread(file, -1), dtype='float32') for file in files_seq]
    else:
        images = [np.array(cv2.imread(file, -1)) for file in files_seq]

    np.save(os.path.join(PATH_TO_NUMPY_SEQ, filename), images)
    np.save(os.path.join(PATH_TO_NUMPY_LABEL, filename), illuminant)


def convert_data(file_names: list, set_type: str):
    """
    Fetches the raw dataset items from the dataset and convert them into Numpy binary files
    @param file_names: the list of names of files to be processed
    @param set_type: the type of set being processed (in {"train", "test"})
    """
    print("\n Length of {} set: {} \n".format(set_type, len(file_names)))

    for i, seq in enumerate(file_names):
        illuminant = list(map(float, open(os.path.join(seq, GROUND_TRUTH_FILE), "r").readline().strip().split(',')))

        files_seq = glob.glob(os.path.join(seq, "[0-9]*.png"))
        files_seq.sort(key=lambda x: x[:-4].split(os.sep)[-1])

        if TRUNCATE:
            files_seq = files_seq[len(files_seq) - SUBSEQUENCE_LEN:]

        if DOWN_SAMPLE:
            files_seq = np.array(files_seq)[[0, len(files_seq) // 2, -1]]

        seq_id = (i + 1) if USE_CV_METADATA else seq.split(os.sep)[-1]

        print("[ Set type: {} | Item ID: {} | Seq: {} | Len: {}]".format(set_type, seq_id, seq, len(files_seq)))
        save_sequence(files_seq, illuminant, filename="{}{}".format(set_type, seq_id))


if __name__ == '__main__':
    main()
