# Experiments And Ablation Notes

This project compares multiple 2D U-Net training setups for binary brain tumor segmentation.

## Experiment Summary

| Model | Checkpoint | Loss Function | Main Metric | Reported Test Dice |
| --- | --- | --- | --- | --- |
| Model 1 | `2D_secondrun.hdf5` | Binary cross-entropy | Accuracy + Dice | 0.76 |
| Model 2 | `model2_1.hdf5` | Dice loss + weighted BCE | Dice score | 0.756 |
| Model 3 | `model2.hdf5` | Tuned Dice loss + weighted BCE | Dice score | 0.815 |

## Why Dice Score Matters

Brain tumor segmentation is class-imbalanced. Most pixels belong to the background, while tumor pixels occupy a much smaller region. A model can therefore achieve high accuracy while still producing weak tumor masks.

Dice coefficient is more useful because it measures overlap between the predicted mask and the ground-truth mask.

## Model Comparison

### Model 1: Binary Cross-Entropy Baseline

The first setup used binary cross-entropy and accuracy-oriented monitoring. It produced high accuracy, but accuracy is not enough for segmentation because background pixels dominate the image.

### Model 2: Dice + Weighted BCE

The second setup introduced a custom objective combining Dice loss and weighted binary cross-entropy. This was designed to make the model more sensitive to tumor pixels.

### Model 3: Tuned Dice-Aware Training

The final reported setup used the Dice-aware objective with tuned training parameters. This produced the strongest reported Dice score of approximately `0.815`.

## Error Analysis

Common segmentation challenges in this task include:

- Very small tumor regions are harder to localize.
- Boundary pixels can be ambiguous around tumor edges.
- False positives may appear around visually similar high-intensity regions.
- 2D slice-based models do not use full 3D context between adjacent MRI slices.
- Dice score is sensitive to small absolute errors when the tumor region is tiny.

## Takeaway

The strongest result came from optimizing for segmentation overlap rather than plain classification accuracy. This is the central ML lesson from the project: metric and loss design must match the structure of the task.
