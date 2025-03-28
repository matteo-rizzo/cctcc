# Cascading Convolutional Colour Constancy

## Paper

Matteo Rizzo, Cristina Conati, Daesik Jang, and Hui Hu "Cascading convolutional temporal color constancy," Journal of Electronic Imaging 32(1), 013049 (28 February 2023). https://doi.org/10.1117/1.JEI.32.1.013049

## Installation

This implementation uses [Pytorch](http://pytorch.org/). It was developed and tested using torch >= 1.7.1 and python 3.6
on Ubuntu 18.04.

Requirements can be installed using the following command:

```shell
pip install -r requirements.txt
```

## Dataset

The TCC dataset was introduced in the paper [A Benchmark for Temporal Color Constancy](https://arxiv.org/abs/2003.03763)
and can be downloaded and preprocessed following the instructions reported in the
related [code repository](https://github.com/yanlinqian/Temporal-Color-Constancy). A polished version of the code for
processing the sequences is available at `dataset/tcc/img2npy.py` in the present repository. This script generates the
preprocessed data at `dataset/tcc/preprocessed` and can be used to generate the splits for a 3-folds CV based on a CSV
file containing the metadata (i.e., which items belong to which split). The metadata for the CV can be generated running
the `dataset/tcc/cv_split.py` script and the file needed to replicate the splits used in the experiments is provided
as `dataset/tcc/3_folds_experiment.csv`. The preprocessed sequences must be placed in `dataset/tcc/preprocessed` with
respect to the root folder.

The script for preprocessing can be run with `python3 img2npy.py` from within the `dataset/tcc/` folder.

The  `dataset/tcc/img2npy.py` file contains the following global variables that can be edited to configured the
preprocessing of the data (note that all paths are relative to `dataset/tcc`).

```python
# Whether or not to use the CV metadata file for generating the preprocessed files
USE_CV_METADATA = False

# Whether or not the CV metadata contains a validation set
USE_VAL_SET = False

# The id of the CV split to be generated (i.e, the corresponding index in the CSV with the metadata) 
FOLD_NUM = 0

# The name of the CSV file containing the metadata for the CV
CV_METADATA_FILE = "3_folds_experiment.csv"

# Whether or not to save data using float64 (results in a relevant increase in space disk required)
USE_HIGH_PRECISION = False

# Whether or not to truncate the sequences keeping only the last SUBSEQUENCE_LEN frames
TRUNCATE = False

# The number of frames to be kept in the truncation process, adjacent to the shot frame (e.g., 2 means shot and preceding)
SUBSEQUENCE_LEN = 2

# Whether or not to down sample the sequences taking the first, median and shot frame only (note that this operation follows the truncation in the preprocessing pipeline)
DOWN_SAMPLE = False

# Base path to the folder containing the preprocessed data
BASE_PATH_TO_DATA = os.path.join("preprocessed", "fold_" + str(FOLD_NUM) if USE_CV_METADATA else "tcc_split")

# Path to the folder containing the sequences in numpy format
PATH_TO_NUMPY_SEQ = os.path.join(BASE_PATH_TO_DATA, "ndata_seq")

# Path to the folder containing the labels (i.e., ground truth illuminants) in numpy format
PPATH_TO_NUMPY_LABEL = os.path.join(BASE_PATH_TO_DATA, "nlabel")

# The base path to the raw sequences to be preprocessed
BASE_PATH_TO_DATASET = "raw"

# The path to the raw training set to be preprocessed
PATH_TO_TRAIN = os.path.join(BASE_PATH_TO_DATASET, "train")

# The path to the raw test set to be preprocessed
PATH_TO_TEST = os.path.join(BASE_PATH_TO_DATASET, "test")

# The name of the file containing the groundtruth of a sequence (located at, e.g., "raw/train/1/")
GROUND_TRUTH_FILE = "groundtruth.txt"
```

## Pretrained models

Pretrained models in PTH format can be downloaded
from [here](https://drive.google.com/drive/folders/1WCOL7Ahy_tDCQDvqn0KQqRhv7eukf3jf?usp=sharing). Please contact me if you have any trouble downloading the models and I will be happy to figure out a way to deliver
them to you.

The pretrained models are organized based on the type of sequences they have been trained on, that is full
sequences (`full_seq`) sequences of 3 frames (`3f_seq`) and sequences of 2 frames (`2f_seq`). All models come with a
dump of the network architecture and of the training metrics. To reproduce the results reported in the paper, the path
to the pretrained models must be specified in the corresponding testing script.

## Structure of the project

The code in this project is mainly structured following object-oriented programming. The core code for the project is
stored under `classes` . The implemented neural network architectures are located at `classes/modules` and organized in
the `multiframe` folder, including [TCCNet](https://github.com/yanlinqian/Temporal-Color-Constancy), TCCNet-C4, C-TCCNet
and C-TCCNet-C4. The code for TCCNet-C4 also includes a reimplementation of its submodules (
i.e., [C4](https://github.com/yhlscut/C4) and [FC4](https://github.com/yuanming-hu/fc4)).

Each module at `classes/modules` features a **network** (i.e., a subclass of `nn.Module`) and a **model** (i.e., a
subclass of the custom `classes/modules/common/BaseModel.py` handling the prediction step). Note that each
model *[has a](https://en.wikipedia.org/wiki/Has-a)* network and acts as interface towards it for training and
inference, handling the weights update and the loss computation.

The `auxiliary/utils.py` file features two functions:

* `get_device`: instantiates the Torch DEVICE (i.e., either CPU or GPU) for training and testing. The DEVICE type can be
  edited to the corresponding global variable at the top of the file.
* `make_deterministic`: sets the random seed. Note that models have been trained using a mix of Tesla P100 and NVidia
  GeForce GTX 1080 Ti GPUs from local lab equipment and cloud services. Please refer to
  the [official PyTorch docs](https://pytorch.org/docs/stable/notes/randomness.html) for an explanation on how
  randomness is handled.

## Running the code

### Training

**TCCNet** and **TCCNet-C4** can be trained with the `python3 train/tcc/train_tccnet.py` command. The file includes the
following global variables that can be edited to configure the training:

```python
# The model to be trained (i.e., either "tccnet" or "tccnetc4")
MODEL_TYPE = "tccnet"

# The folder at "dataset/tcc/preprocessed" containing the data the model must be trained on 
DATA_FOLDER = "tcc_split"

# Whether or not to resume the training based on an existing checkpoint model in PTH format
RELOAD_CHECKPOINT = False

# The path to the PTH file to be used as checkpoint
PATH_TO_PTH_CHECKPOINT = os.path.join("trained_models", "tccnetc", "checkpoint", "model.pth")

# The number of iterations over data
EPOCHS = 2000

# The size of the batch to be fed to the network (only 1 is currently supported) 
BATCH_SIZE = 1

# The learning rate to which the optimizer must be initialized
LEARNING_RATE = 0.00003
```

**C-TCCNet** and **C-TCCNet-C4** can be trained with the `python3 train/tcc/train_ctccnet/.py` command. The file
includes the following global variables that can be edited to configure the training:

```python
# The model to be trained (i.e., either "ctccnet" or "ctccnetc4")
MODEL_TYPE = "ctccnet"

# The folder at "dataset/tcc/preprocessed" containing the data the model must be trained on 
DATA_FOLDER = "tcc_split"

# The path to the pretrained submodule in PTH format (i.e., TCCNet for C-TCCNet and TCCNet-C4 for C-TCCNet-C4)
PATH_TO_PTH_SUBMODULE = os.path.join("trained_models", "tccnet", "model.pth")

# Whether or not to resume the training based on an existing checkpoint model in PTH format
RELOAD_CHECKPOINT = False

# The path to the PTH file to be used as checkpoint
PATH_TO_PTH_CHECKPOINT = os.path.join("trained_models", "ctccnet", "checkpoint", "model.pth")

# The number of iterations over data
EPOCHS = 2000

# The size of the batch to be fed to the network (only 1 is currently supported) 
BATCH_SIZE = 1

# The learning rate to which the optimizer must be initialized
LEARNING_RATE = 0.00003
```

### Testing

**TCCNet** and **TCCNet-C4** can be tested to replicate the results reported in the paper with
the `python3 test/tcc/test_tccnet.py` script and loading the desired pretrained model. The file includes the following
global variables that can be edited to configure the training:

```python
# The model to be trained (i.e., either "tccnet" or "tccnetc4")
MODEL_TYPE = "tccnet"

# The folder at "dataset/tcc/preprocessed" containing the data the model must be tested on 
DATA_FOLDER = "tcc_split"

# The path to the pretrained PTH model to be loaded for testing
PATH_TO_PTH = os.path.join("trained_models", "tccnet", "model.pth")
```

**C-TCCNet** and **C-TCCNet-C4** can be tested to replicate the results reported in the paper with
the `python3 test/tcc/test_ctccnet.py` command and loading the desired pretrained model. The file includes the following
global variables that can be edited to configure the training:

```python
# The model to be trained (i.e., either "ctccnet" or "ctccnetc4")
MODEL_TYPE = "ctccnet"

# The folder at "dataset/tcc/preprocessed" containing the data the model must be tested on 
DATA_FOLDER = "tcc_split"

# The path to the pretrained PTH model to be loaded for testing
PATH_TO_PTH = os.path.join("trained_models", "ctccnet", "model.pth")
```

## Cite

```latex
@article{10.1117/1.JEI.32.1.013049,
  author = {Matteo Rizzo and Cristina Conati and Daesik Jang and Hui Hu},
  title = {{Cascading convolutional temporal color constancy}},
  volume = {32},
  journal = {Journal of Electronic Imaging},
  number = {1},
  publisher = {SPIE},
  pages = {013049},
  keywords = {color constancy, videos, sequences, Performance modeling, Current controlled current source, Data modeling, Education and training, Video, Color, Light sources and illumination, Network architectures, RGB color model, Video surveillance},
  year = {2023},
  doi = {10.1117/1.JEI.32.1.013049},
  URL = {https://doi.org/10.1117/1.JEI.32.1.013049}
}
```
