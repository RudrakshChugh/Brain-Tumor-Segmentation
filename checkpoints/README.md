# Model Checkpoints

This folder contains saved Keras/TensorFlow model checkpoints from different 2D U-Net training runs.

The checkpoints are included so the trained segmentation models can be reused for inference, comparison, or inspection without retraining from scratch.

## Checkpoint Guide

| File | Description |
| --- | --- |
| `2D_firstrun.hdf5` | Early 2D U-Net training run. |
| `2D_secondrun.hdf5` | Baseline 2D U-Net run using binary cross-entropy and accuracy-oriented monitoring. |
| `best_model.h5` | Best saved model artifact from one training experiment. |
| `best_model2.h5` | Best saved model artifact from a later or alternate training experiment. |
| `model2_1.hdf5` | 2D U-Net trained with a custom Dice + weighted binary cross-entropy objective. |
| `model2.hdf5` | Tuned 2D U-Net checkpoint associated with the strongest reported Dice score. |

## How These Fit Into The Project

The project compares multiple segmentation training setups:

1. A baseline U-Net trained with binary cross-entropy.
2. A Dice-aware U-Net trained with Dice loss and weighted binary cross-entropy.
3. A tuned Dice-aware U-Net with improved reported segmentation performance.

The final reported model achieved approximately:

```text
Test Dice Score: 0.815
```

## Loading A Checkpoint

Depending on whether the checkpoint stores a full model or weights only, it can be loaded with one of the following Keras patterns.

Load a full saved model:

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "checkpoints/model2.hdf5",
    compile=False,
)
```

Load weights into the project model architecture:

```python
import tensorflow as tf
from src.model import create_unet

inputs = tf.keras.layers.Input((128, 128, 3))
model = create_unet(inputs)
model.load_weights("checkpoints/model2.hdf5")
```

If a checkpoint was trained with custom metrics or custom loss functions, pass those objects through `custom_objects` when loading with compilation enabled.

## Notes

- These files are binary model artifacts, so their contents are not human-readable like source code.
- Their purpose is to preserve trained model states from experiments.
- The source code in `src/` explains how the models are built, trained, and evaluated.
