import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def predict_test_samples(X_test, y_test, model):
    """Run model inference on a test batch and return predictions with inputs."""
    test_images = np.array(X_test)
    predictions = model.predict(test_images)
    return predictions, test_images, y_test

def plot_images(test_image, predicted_mask, ground_truth):
    """Plot an input image, predicted mask, and ground-truth mask side by side."""
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.imshow(test_image)
    plt.title('Image')
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask)
    plt.title('Predicted mask')
    plt.subplot(1, 3, 3)
    plt.imshow(ground_truth)
    plt.title('Ground truth mask')

def dice_coefficient(ground_truth_masks, predicted_masks, smooth=1):
    """Compute the mean Dice coefficient for batches of binary masks."""
    intersection = tf.reduce_sum(ground_truth_masks * predicted_masks, axis=[1, 2, 3])
    union = tf.reduce_sum(ground_truth_masks, axis=[1, 2, 3]) + tf.reduce_sum(predicted_masks, axis=[1, 2, 3])
    dice = tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def process_predictions(predicted_masks, threshold=0.5):
    """Convert probability masks to binary masks using a threshold."""
    return (predicted_masks > threshold).astype(np.float32)
