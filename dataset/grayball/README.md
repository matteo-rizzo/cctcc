# SFU Gray Ball Dataset

* **Paper**: https://www2.cs.sfu.ca/~colour/publications/PCIC-2003/LargeImageDatabase.pdf
* **Requesting the data:** https://www2.cs.sfu.ca/~colour/data/gray_ball/

## Specifics

The Gray Ball was generated from 15 video clips with significantly different content. From each video, 81 − 1312 frames
were selected and provided with ground truth. The videos are sampled at roughly 3 frames per second. The images in the
original dataset are stored in a non-linear DEVICE specific RGB color space.

The .zip file contains:

1. The images, and
2. A file with the image names in the order corresponding to the .mat file.

The .mat file is a Matlab matrix file containing an 11346 by 3 matrix. Each row is the RGB of illuminant for the
corresponding file in file.lst.

In the 11346 dataset includes 9 images that aren't in the file.lst or .mat files. They're just extra and they don't
affect any of the results. However, in case you wonder about them there are 9 of them which are extra:

* raw/Burnaby_Mountain/01270.jpg
* raw/CIC2002/05860.jpg
* raw/CIC2002/11740.jpg
* raw/CIC2002/03110.jpg
* raw/CIC2002/05570.jpg
* raw/Granville_Island_Market2/07670.jpg
* raw/Granville_Island_Market2/10660.jpg
* raw/Metrotown/16980.jpg
* raw/SFU/00670.jpg

**Note**: for practical convenience, these frames have been physically removed from the dataset to generate the
sequences for the temporal scenario.

## Preprocessing

1. **Extract subsequences.**
    1. All sequences have fixed length *N*. The paper ["Recurrent Color Constancy"](https://openaccess.thecvf.com/content_ICCV_2017/papers/Qian_Recurrent_Color_Constancy_ICCV_2017_paper.pdf) suggests *N = 5* as best trade-off between accuracy and required resources.
    2. For each scene, given the ordered sequence of frames `f_0, f_1, ..., f_n`, no sequences is generated for the
       first *N - 1* frames (as the would not have enough preceding frames for the temporal processing). For example,
       if *N = 5*, frames `f_0, f_1, f_2, f_3` are skipped
2. **Data augmentation**. Performed as for Temporal Color Constancy (TCC) Dataset:
    1. Random rotation in [-30°, +30]
    2. Random crop in scale [0.8, 1.0] on the shorter dimension
    3. Random horizontal flip with probability *p = 0.5*
    2. Gamma-correction (for *gamma = 2.2*)

## Benchmark protocol

15-fold cross-validation by leave-one-sequence-out. Sequence border effects were handled by repetition of the first
frame in a video. Since predictions for all frames of the videos were made, the proposed method can be compared with
single-frame methods. Consistent with the preprocessing of the ordinary Gray Ball Dataset, the pixels of the gray
sphere, which is in a known fixed location, are excluded by cropping.

## Citing the dataset

> Ciurea, F. and Funt, B. "A Large Image Database for Color Constancy Research," Proceedings of the Imaging Science and Technology Eleventh Color Imaging Conference, pp. 160-164, Scottsdale, Nov. 2003.

In terms of publishing sample images, the only restriction is that you not publish any that show people's faces.