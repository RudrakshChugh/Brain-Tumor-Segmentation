# Model Card: Brain Tumor MRI Segmentation U-Net

## Model Summary

This repository contains U-Net based segmentation models for identifying tumor regions in brain MRI images. The primary completed model is a 2D binary segmentation model trained on MRI images and corresponding tumor masks.

## Intended Use

- Educational demonstration of medical image segmentation
- ML portfolio project
- Experimentation with U-Net, Dice loss, and segmentation metrics
- Research-style exploration of 2D versus 3D segmentation constraints

## Not Intended For

- Clinical diagnosis
- Treatment planning
- Automated medical decision-making
- Use without expert validation

## Input

The main 2D model expects RGB images resized to:

```text
128 x 128 x 3
```

Pixel values are normalized to `[0, 1]`.

## Output

The model outputs a binary probability mask:

```text
128 x 128 x 1
```

Predictions can be thresholded at `0.5` to produce binary segmentation masks.

## Evaluation Metric

The main evaluation metric is Dice coefficient, which measures overlap between predicted and ground-truth masks.

Dice is preferred because tumor segmentation is class-imbalanced: most pixels are background, so accuracy can look high even when tumor localization is weak.

## Reported Performance

The best reported 2D model achieved approximately:

```text
Test Dice Score: 0.815
```

## Limitations

- Performance depends on dataset quality and preprocessing consistency.
- The 2D model does not use full volumetric context between MRI slices.
- The 3D U-Net path was constrained by available GPU memory.
- External validation on independent clinical datasets is not included.
- The model should not be interpreted as medically reliable.

## Ethical And Safety Notes

Medical AI systems can produce incorrect or misleading predictions. This project is a learning artifact and must not be used for patient care. Any medical imaging model requires clinical validation, expert review, bias analysis, and deployment safeguards before real-world use.
