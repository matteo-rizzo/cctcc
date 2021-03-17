import os
from glob import glob

import numpy as np
from tqdm import tqdm

PATH_TO_DATA = os.path.join("raw")


def main():
    print("\n=================================================\n")
    print("\t TCC Dataset Exploration")
    print("\n=================================================\n")

    path_to_train = os.path.join(PATH_TO_DATA, "train")
    paths_to_train = [os.path.join(path_to_train, seq) for seq in os.listdir(path_to_train)]

    path_to_test = os.path.join(PATH_TO_DATA, "test")
    paths_to_test = [os.path.join(path_to_test, seq) for seq in os.listdir(path_to_test)]

    paths_to_seq = paths_to_train + paths_to_test

    seq_lengths = np.array([len(glob(os.path.join(path_to_seq, "[0-9]*.png"))) for path_to_seq in tqdm(paths_to_seq)])

    print(seq_lengths)
    print("Lengths: [ Avg: {} | Std dev: {} ]".format(seq_lengths.mean(), seq_lengths.std()))

    print("\n=================================================\n")
    print("\t Exploration completed successfully!")
    print("\n=================================================\n")


if __name__ == '__main__':
    main()
