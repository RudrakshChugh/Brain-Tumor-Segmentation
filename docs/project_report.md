# Project Report: Brain Tumor Segmentation from MRI using U-Net

## Overview

This project implements a deep learning pipeline for brain tumor segmentation from MRI scans. The primary completed workflow uses a 2D U-Net architecture to predict binary tumor masks from MRI images. The project also explores a 3D U-Net approach for volumetric multi-modal MRI data, but full 3D training was constrained by local GPU memory.

## Motivation

Medical image segmentation is a key task in computer vision for healthcare. In brain MRI analysis, segmentation can help isolate regions of interest and support quantitative study of tumor shape, size, and location. Tumor segmentation is difficult because the foreground class is often small compared with the background, making accuracy alone a misleading metric.

## Data

The main 2D pipeline uses the LGG MRI Segmentation Dataset. Each sample consists of an MRI image and a corresponding manual segmentation mask. Images and masks are resized to `128 x 128`. Pixel values are normalized to `[0, 1]`, and masks are converted to binary values.

The 3D exploration uses BraTS-style multi-modal MRI volumes. These contain multiple MRI channels and multi-class labels, making the task more expressive but significantly more expensive to train.

## Pipeline

1. Load MRI image and mask paths.
2. Read images using OpenCV.
3. Resize images and masks to a common resolution.
4. Normalize images.
5. Convert masks to binary tensors.
6. Train a U-Net segmentation model.
7. Optimize with a Dice-aware objective.
8. Generate predicted masks.
9. Evaluate using Dice coefficient.
10. Export prediction visualizations.

## Model

The main model is a U-Net style encoder-decoder network. The encoder captures increasingly abstract visual features through convolutional blocks and pooling. The decoder upsamples these features back to image resolution. Skip connections copy spatial information from encoder layers into corresponding decoder layers, improving boundary localization.

The output layer uses a sigmoid activation to produce a probability mask for tumor pixels.

## Loss And Metrics

The project uses Dice score as the main segmentation metric. Dice is better suited than plain accuracy for imbalanced segmentation because it directly measures overlap between predicted and ground-truth masks.

The stronger training setup combines:

- Dice loss
- Weighted binary cross-entropy
- Adam optimizer with cosine learning-rate decay

This objective helps the model pay more attention to tumor pixels despite background dominance.

## Results

| Model | Training Setup | Reported Test Dice |
| --- | --- | --- |
| Model 1 | Binary cross-entropy baseline | 0.76 |
| Model 2 | Dice + weighted BCE | 0.756 |
| Model 3 | Tuned Dice + weighted BCE | 0.815 |

The final reported model achieved approximately `0.815` test Dice score. Prediction images are archived in `results/predicted_result.zip`, with selected samples available in `results/sample_predictions/`.

## 3D U-Net Exploration

The repository includes utilities for 3D visualization, volumetric data loading, and 3D U-Net model construction. Full 3D training was attempted but limited by GPU memory. This is a realistic constraint in volumetric medical imaging, where input tensors are much larger than 2D slices.

## Key Learnings

- U-Net is effective for biomedical image segmentation because it combines semantic context with spatial precision.
- Dice coefficient is more informative than accuracy for imbalanced masks.
- Weighted losses can help with tumor-background imbalance.
- 3D medical imaging models require careful memory planning.
- Packaging checkpoints, results, and documentation makes ML work easier to review and reproduce.

## Responsible Use

This project is educational and experimental. It is not validated for clinical use and should not be used for diagnosis or treatment planning.
