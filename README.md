# Brain Tumor Segmentation from MRI using U-Net

An end-to-end deep learning project for segmenting brain tumor regions from MRI scans using U-Net based convolutional neural networks.

The project focuses on medical image segmentation, custom loss design for imbalanced masks, Dice-based evaluation, and practical model packaging with trained checkpoints and prediction artifacts.

## Highlights

- Built a modular 2D MRI segmentation pipeline using TensorFlow/Keras.
- Implemented U-Net encoder-decoder models with skip connections.
- Added custom Dice score metric and combined Dice + weighted binary cross-entropy loss.
- Evaluated segmentation quality using Dice coefficient rather than accuracy alone.
- Packaged trained model checkpoints and prediction visualizations.
- Explored a 3D U-Net path for volumetric MRI segmentation and documented hardware constraints.
- Added checkpoint-based inference, experiment metrics, and dataset setup documentation for reproducibility.

## Problem Statement

Brain tumor segmentation aims to identify tumor regions at the pixel level from MRI scans. This is a challenging medical imaging task because tumor regions are often small, irregularly shaped, and highly imbalanced compared with the background.

This project uses semantic segmentation to produce a binary mask for each MRI image:

- Input: brain MRI image
- Output: tumor segmentation mask

## Results

The best reported model used a tuned Dice-aware objective and achieved the strongest segmentation performance.

| Model | Loss / Objective | Main Metric | Reported Test Dice |
| --- | --- | --- | --- |
| Model 1 | Binary cross-entropy | Accuracy + Dice | 0.76 |
| Model 2 | Dice loss + weighted BCE | Dice score | 0.756 |
| **Model 3** | **Tuned Dice loss + weighted BCE** | **Dice score** | **0.815** |

Accuracy was high for early models, but Dice coefficient is the more meaningful metric for this task because the tumor region occupies a much smaller area than the background.

The metrics are also stored in `results/metrics.json` for programmatic inspection.

### Sample Predictions

Sample prediction artifacts from the final 2D model output:

| Prediction 1 | Prediction 2 |
| --- | --- |
| ![Prediction 0](results/sample_predictions/predicted_0.jpg) | ![Prediction 10](results/sample_predictions/predicted_10.jpg) |

| Prediction 3 | Prediction 4 |
| --- | --- |
| ![Prediction 50](results/sample_predictions/predicted_50.jpg) | ![Prediction 100](results/sample_predictions/predicted_100.jpg) |

The complete prediction archive is available at `results/predicted_result.zip`.

## Datasets

| Dataset | Source | Usage |
| --- | --- | --- |
| **LGG MRI Segmentation** | Kaggle, `mateuszbuda/lgg-mri-segmentation` | Main 2D U-Net pipeline. Contains MRI images and manually annotated FLAIR abnormality segmentation masks. Images and masks are resized to `128 x 128`. |
| **BraTS 2020** | Kaggle, `awsaf49/brats20-dataset-training-validation` | 3D U-Net exploration. Contains multi-modal volumetric MRI scans (T1, T1ce, T2, FLAIR). Full 3D training was limited by local GPU memory constraints. |

## Methodology

```text
MRI Data
  -> Image and mask loading
  -> Resizing and normalization
  -> U-Net segmentation model
  -> Dice + weighted BCE optimization
  -> Validation and checkpointing
  -> Prediction mask generation
  -> Dice coefficient evaluation
```

## Model Architecture

The main model is a 2D U-Net:

| Stage | Description |
| --- | --- |
| **Encoder** | Repeated convolution blocks with batch normalization and ReLU activations |
| **Downsampling** | Max pooling and dropout |
| **Bottleneck** | Deeper convolutional feature extraction |
| **Decoder** | Transposed convolutions for upsampling |
| **Skip Connections** | Concatenate encoder features with decoder features |
| **Output** | `1 x 1` convolution with sigmoid activation for binary segmentation |

The repository also includes a 3D U-Net implementation for volumetric MRI experiments.

## Repository Structure

```text
├── configs/
│   └── unet_2d.yaml             # Main 2D U-Net experiment configuration
├── checkpoints/
│   ├── 2D_firstrun.hdf5
│   ├── 2D_secondrun.hdf5
│   ├── best_model.h5
│   ├── best_model2.h5
│   ├── model2.hdf5
│   ├── model2_1.hdf5
│   └── README.md
├── docs/
│   ├── dataset_setup.md          # Dataset download & folder-structure guide
│   ├── experiments.md            # Experiment comparison & ablation notes
│   ├── model_card.md             # Intended use, limitations & responsible-use notes
│   └── project_report.md         # Detailed project writeup
├── outputs/                      # Inference output directory
├── results/
│   ├── metrics.json              # Model comparison metrics
│   ├── predicted_result.zip      # Full prediction archive
│   └── sample_predictions/       # Selected prediction visualizations
├── src/
│   ├── dataset.py                # Data loading & preprocessing
│   ├── evaluate.py               # Dice coefficient & evaluation logic
│   ├── model.py                  # U-Net architecture definition
│   ├── predict.py                # Checkpoint-based inference
│   ├── train.py                  # Training loop & checkpointing
│   └── utils.py                  # Shared utilities
├── requirements.txt
└── README.md
```

## Setup & Quick Start

### 1. Environment Configuration

```bash
pip install -r requirements.txt
```

The project was built around TensorFlow/Keras, OpenCV, NumPy, Matplotlib, h5py, and supporting medical-imaging libraries.

### 2. Training

The source modules are organized so the workflow can be used from scripts or notebooks.

Load 2D image and mask paths:

```python
from src.dataset import get_images_path, read_images

image_paths, mask_paths = get_images_path("path/to/lgg-mri-segmentation")
images = read_images(image_paths, file_type="images")
masks = read_images(mask_paths, file_type="masks")
```

Train the 2D U-Net:

```python
from src.train import train_model

model, history = train_model(X_train, y_train, X_val, y_val)
```

### 3. Evaluation

```python
from src.evaluate import dice_coefficient, process_predictions

binary_predictions = process_predictions(predicted_masks)
dice = dice_coefficient(y_test, binary_predictions)
```

### 4. Inference

Run prediction with a trained checkpoint:

```bash
python src/predict.py \
  --image path/to/input_mri.png \
  --checkpoint checkpoints/model2.hdf5 \
  --output-dir outputs
```

The script saves:

- `outputs/predicted_mask.png`
- `outputs/prediction_overlay.png`

## Project Documents

| Document | Description |
| --- | --- |
| `configs/unet_2d.yaml` | Main 2D U-Net experiment configuration |
| `results/metrics.json` | Reported model comparison metrics |
| `docs/dataset_setup.md` | Dataset download and folder-structure guide |
| `docs/experiments.md` | Experiment comparison, ablation notes, and error analysis |
| `docs/project_report.md` | Detailed project writeup |
| `docs/model_card.md` | Model card with intended use, limitations, and responsible-use notes |
| `checkpoints/README.md` | Explanation of saved model checkpoints and experiment mapping |

## Responsible Use

This project is for learning and portfolio demonstration only. It is not a clinical diagnostic system and should not be used for medical decision-making.
