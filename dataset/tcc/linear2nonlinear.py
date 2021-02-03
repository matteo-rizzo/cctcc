import glob
import os

from PIL import Image
from tqdm import tqdm

from viz.utils import linear_to_nonlinear

NUM_EXAMPLES = 25

DATA_TYPE = "test"
PATH_TO_DATA = os.path.join("raw", DATA_TYPE)

PATH_TO_RESULTS = os.path.join("non_linear", DATA_TYPE)


def main():
    print("\n=================================================\n")
    print("\t TCC sequences Linear to Non-linear")
    print("\n=================================================\n")

    os.makedirs(PATH_TO_RESULTS, exist_ok=True)

    for i, seq_id in tqdm(enumerate(os.listdir(PATH_TO_DATA))):

        if NUM_EXAMPLES != -1 and i >= NUM_EXAMPLES:
            break

        path_to_seq = os.path.join(PATH_TO_DATA, seq_id)
        paths_to_frames = glob.glob(os.path.join(path_to_seq, "[0-9]*.png"))
        paths_to_frames.sort(key=lambda x: x[:-4].split(os.sep)[-1])

        path_to_result = os.path.join(PATH_TO_RESULTS, seq_id)
        os.makedirs(path_to_result)

        for path_to_frame in paths_to_frames:
            frame = linear_to_nonlinear(Image.open(path_to_frame))
            frame.save(os.path.join(path_to_result, path_to_frame.split(os.sep)[-1]))

        gt = linear_to_nonlinear(Image.open(os.path.join(path_to_seq, "groundtruth.png")))
        gt.save(os.path.join(path_to_result, "groundtruth.png"))

    print("\n=================================================\n")
    print("\t Sequences processed successfully!")
    print("\n=================================================\n")


if __name__ == '__main__':
    main()
