import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

try:
    from model import create_unet
except ImportError:
    from .model import create_unet


IMAGE_SIZE = (128, 128)


def load_image(image_path):
    """Load and normalize an MRI image for 2D U-Net inference."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    original = image.copy()
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype(np.float32) / 255.0
    return original, np.expand_dims(image, axis=0)


def load_checkpoint(checkpoint_path):
    """Load a Keras checkpoint for inference."""
    checkpoint_path = str(checkpoint_path)

    try:
        return tf.keras.models.load_model(checkpoint_path, compile=False)
    except Exception:
        inputs = tf.keras.layers.Input((128, 128, 3))
        model = create_unet(inputs)
        model.load_weights(checkpoint_path)
        return model


def predict_mask(model, image_batch, threshold):
    """Predict and threshold a binary segmentation mask."""
    probability_mask = model.predict(image_batch)[0]
    binary_mask = (probability_mask > threshold).astype(np.uint8) * 255
    return np.squeeze(binary_mask)


def create_overlay(original_image, mask):
    """Create a red mask overlay on top of the original image."""
    resized_mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
    overlay = original_image.copy()
    red_layer = np.zeros_like(original_image)
    red_layer[:, :, 2] = resized_mask
    return cv2.addWeighted(overlay, 0.75, red_layer, 0.25, 0)


def run_inference(image_path, checkpoint_path, output_dir, threshold):
    """Run checkpoint-based inference and save mask plus overlay artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    original_image, image_batch = load_image(image_path)
    model = load_checkpoint(checkpoint_path)
    mask = predict_mask(model, image_batch, threshold)
    overlay = create_overlay(original_image, mask)

    mask_path = output_dir / "predicted_mask.png"
    overlay_path = output_dir / "prediction_overlay.png"

    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(overlay_path), overlay)

    return mask_path, overlay_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run brain tumor segmentation inference with a trained U-Net checkpoint.")
    parser.add_argument("--image", required=True, type=Path, help="Path to an input MRI image.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to a .h5 or .hdf5 Keras checkpoint.")
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path, help="Directory for predicted outputs.")
    parser.add_argument("--threshold", default=0.5, type=float, help="Probability threshold for binary mask generation.")
    return parser.parse_args()


def main():
    args = parse_args()
    mask_path, overlay_path = run_inference(args.image, args.checkpoint, args.output_dir, args.threshold)
    print(f"Saved predicted mask to: {mask_path}")
    print(f"Saved prediction overlay to: {overlay_path}")


if __name__ == "__main__":
    main()
