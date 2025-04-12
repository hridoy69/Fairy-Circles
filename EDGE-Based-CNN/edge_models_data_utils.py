# data_utils.py

import os
import random
import numpy as np


from skimage import (
    io,
    feature,
    color
)

from skimage.filters import (
    roberts,
    sobel,
    scharr, 
    prewitt
)

import tensorflow as tf

# Define default dataset directory
DATASET_DIR = '/data'  # <--- Set your path here
SEED = 42

# Set the image size and batch size
IMAGE_SIZE = (224, 224)  # AlexNet input size
BATCH_SIZE = 32

def set_global_seed(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

## ----------------------------------------------------------------- Custom preprocessing wrapper function -------------------------------------------------------- #
def preprocessing_wrapper(method='canny'):
    def preprocessing_function(image):
        return perform_edge_detection(image, method=method)
    return preprocessing_function

def perform_edge_detection(image,method):
    image = color.rgb2gray(image)    # Convert to grayscale
    
    if method == 'canny':
        image = feature.canny(image, sigma=0.02).astype(np.float32)
    if method == 'sobel':
        image = sobel(image).astype(np.float32)
    if method == 'roberts':
        image = roberts(image).astype(np.float32)
    if method == 'scharr':
        image = scharr(image).astype(np.float32)
    if method == 'prewitt':
        image = prewitt(image).astype(np.float32)
        
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.repeat(image, 3, axis=-1)    # Repeat the single channel to create a 3-channel image
    return image


