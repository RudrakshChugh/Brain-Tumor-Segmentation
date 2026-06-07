import os
import cv2
import numpy as np
from tqdm import tqdm

def get_images_path(root):
    """Return paired image and mask paths from the LGG MRI dataset layout."""
    images = []
    labels = []
    for d in tqdm(os.listdir(root)):
        path = os.path.join(root, d)
        if os.path.isdir(path):
            iters = int(len(os.listdir(path))/2)
            for i in range(iters): 
                file = os.path.join(path, d) + '_' +str(i+1) + '.tif'
                mask = os.path.join(path, d) + '_' +str(i+1) + '_mask.tif'
                images.append(file)
                labels.append(mask)
    return images, labels

def read_image(path):
    """Read and normalize one MRI image for the 2D U-Net pipeline."""
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x,(128,128))
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    """Read one segmentation mask and convert it to a binary tensor."""
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x,(128,128))
    x = x/255.0
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def read_images(paths, file_type):
    """Read a list of image or mask paths into a NumPy array."""
    images = []
    if file_type == 'images':
        for f in tqdm(paths):
            img = read_image(f)
            images.append(img)    
    else:
        for f in tqdm(paths):
            img = read_mask(f)
            images.append(img)
    imgs1 = np.array(images)
    return imgs1
