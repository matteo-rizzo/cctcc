import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

BASE_PATH_TO_DATASET = "raw"
PATH_TO_TRAIN = os.path.join(BASE_PATH_TO_DATASET, "train")
PATH_TO_TEST = os.path.join(BASE_PATH_TO_DATASET, "test")

NUM_SPLITS = 3
USE_VAL_SET = False


def main():
    print("\n=================================================\n")
    print("\t Generating CV splits")
    print("\n=================================================\n")

    train_data = glob.glob(os.path.join(PATH_TO_TRAIN, "*"))
    test_data = glob.glob(os.path.join(PATH_TO_TEST, "*"))

    data = np.array(train_data + test_data)
    np.random.shuffle(data)

    splits = []
    for i, (train_index, val_test_index) in enumerate(KFold(n_splits=3).split(data)):
        train, val_test = data[train_index], data[val_test_index]

        if USE_VAL_SET:
            val, test = train_test_split(val_test, test_size=.5)
            splits.append({"train": train.tolist(), "val": val.tolist(), "test": test.tolist()})
            print("Split {}: [ train: {} | val: {} | test: {} ]".format(i, len(train), len(val), len(test)))
        else:
            splits.append({"train": train.tolist(), "test": val_test.tolist()})
            print("Split {}: [ train: {} | test: {} ]".format(i, len(train), len(val_test)))

    print("\nChecking sanity of generated splits...\n")
    for i, split in enumerate(splits):
        train_set, test_set = split["train"], split["test"]

        if len([e for e in train_set if e in test_set]):
            print("WARNING: Data leakage in split {}!".format(i))

        for set_type in ["train", "test"]:
            if len(split[set_type]) != len(set(split[set_type])):
                print("WARNING: Split {} contains duplicates in the {} set".format(i, set_type))

    pd.DataFrame(splits).to_csv("{}_folds.csv".format(NUM_SPLITS), index=False)

    print("\n TCC split (400 train, 200 test) overlap: \n")
    train_overlaps, test_overlaps = [], []
    metadata = pd.read_csv("{}_folds.csv".format(NUM_SPLITS), converters={'train': eval, 'val': eval, 'test': eval})
    for i in range(NUM_SPLITS):
        train_split, test_split = metadata.iloc[i]["train"], metadata.iloc[i]["test"]
        train_overlap = (sum([1 for item in train_split if item.split(os.sep)[1] == "train"]) / len(train_split)) * 100
        test_overlap = (sum([1 for item in test_split if item.split(os.sep)[1] == "test"]) / len(test_split)) * 100
        print("\t Split {} [train overlap: {:.2f} | test overlap: {:.2f}]".format(i, train_overlap, test_overlap))
        train_overlaps.append(train_overlap)
        test_overlaps.append(test_overlap)
    print("\n AVG overlap [train: {:.2f} | test: {:.2f}] \n".format(np.mean(train_overlaps), np.mean(test_overlaps)))

    print("\n=================================================\n")
    print("\t Generated CV splits!")
    print("\n=================================================\n")


if __name__ == '__main__':
    main()
