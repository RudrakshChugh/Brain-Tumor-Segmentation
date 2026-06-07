# Dataset Setup

This project does not include the raw MRI datasets because they are large and should be downloaded from their original sources.

## Main 2D Dataset

The primary pipeline uses the LGG MRI Segmentation Dataset:

```text
mateuszbuda/lgg-mri-segmentation
```

Download it from Kaggle and place it under a local dataset directory, for example:

```text
data/
`-- lgg-mri-segmentation/
    |-- TCGA_CS_4941_19960909/
    |   |-- TCGA_CS_4941_19960909_1.tif
    |   |-- TCGA_CS_4941_19960909_1_mask.tif
    |   |-- TCGA_CS_4941_19960909_2.tif
    |   `-- TCGA_CS_4941_19960909_2_mask.tif
    `-- ...
```

The loader expects each patient folder to contain matching image and mask files:

```text
<patient_id>_<slice_number>.tif
<patient_id>_<slice_number>_mask.tif
```

## Preprocessing

The 2D preprocessing pipeline performs:

1. RGB image loading with OpenCV.
2. Image resizing to `128 x 128`.
3. Pixel normalization to `[0, 1]`.
4. Grayscale mask loading.
5. Mask resizing to `128 x 128`.
6. Binary thresholding at `0.5`.
7. Mask reshaping to `128 x 128 x 1`.

## 3D Dataset Exploration

The repository also includes utilities for BraTS-style 3D MRI volumes:

```text
awsaf49/brats20-dataset-training-validation
```

This path was used for architecture exploration and volumetric visualization. Full 3D U-Net training requires significantly more GPU memory than the 2D pipeline.

## Expected Local Paths

Raw data should stay outside version control:

```text
data/
datasets/
```

These folders are ignored by `.gitignore`.
