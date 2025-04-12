import os
import csv
import logging

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.layers import (
    Dense, 
    GlobalAveragePooling2D
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

from rgb_models_data_utils import (
    set_global_seed, 
    get_data_generators
)

set_global_seed(42)

train_generator, val_generator = get_data_generators()


# Log class indices
message = f"Class indices: {train_generator.class_indices}"
logging.basicConfig(
    filename='Class-indices.log',
    #filename='Dataset 2 non_fairy.log',
    # filename='non_fairy.log',
    level=logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.info(message)

# Load the pre-trained Inception V3 model without the top classification layer
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global average pooling layer and a dense output layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)  # Add additional dense layers if needed
predictions = Dense(1, activation='sigmoid')(x)  # Output layer with sigmoid activation for binary classification

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the pre-trained model
for layer in base_model.layers:  
    layer.trainable = False

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Custom callback for logging to CSV
class CSVLogger(Callback):
    def __init__(self, filename):
        super(CSVLogger, self).__init__()
        self.filename = filename
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                logs['loss'],
                logs['accuracy'],
                logs['val_loss'],
                logs['val_accuracy']
            ])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath='./inceptionv3_results/inception_best_weights.tf', monitor='val_loss', save_best_only=True, mode='min',save_format="tf")
csv_logger = CSVLogger('./inceptionv3_results/training_log.csv')

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=100,
    callbacks=[
        #early_stopping,
        checkpoint, 
        csv_logger
        ]
)

# Load the best weights
model.load_weights('./inceptionv3_results/inception_best_weights.tf')

# Evaluate the model
val_loss, val_acc = model.evaluate(val_generator)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_acc}')
