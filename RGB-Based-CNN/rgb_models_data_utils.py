# data_utils.py

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

def get_data_generators(dataset_dir = DATASET_DIR, image_size = IMAGE_SIZE, batch_size=BATCH_SIZE, seed=SEED, class_mode='binary'):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='training',
        shuffle=True,
        seed=seed
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='validation',
        shuffle=False,
        seed=seed
    )

    return train_gen, val_gen
