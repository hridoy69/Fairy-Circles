import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import os,csv,logging

############################################################# - START - SAME CODE FOR ALL - ########################################################################

## --------------------------------------------------------------------- Defining Edge Detection Method -------------------------------------------------------- #
epochs = 100
model='alexnet'
method='sobel'

# ----------------------------------------------------- Output directory for saving weights -------------------------------------------------------------- #

output_dir = f'{model}_{method}_results'
os.makedirs(output_dir, exist_ok=True)

## ------------------------------------------------------------------- Setting image size and batch size -------------------------------------------------------- #

# Set the image size and batch size
image_size = (224, 224)  
batch_size = 32

## ----------------------------------------------------------------- Importing Edge Detecting methods -------------------------------------------------------- #
import numpy as np
from skimage import io
from skimage import feature, color
from skimage.filters import roberts, sobel, scharr, prewitt

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



## ------------------------------------------------------------ Defining training and validation paths to dataset -------------------------------------------------------- #

train_dir = os.path.abspath('./FC DATASET/Training set')
val_dir = os.path.abspath('./FC DATASET/Validation set')


train_datagen = ImageDataGenerator(rescale=1.0/255.0,preprocessing_function=preprocessing_wrapper(method))
val_datagen = ImageDataGenerator(rescale=1.0/255.0,preprocessing_function=preprocessing_wrapper(method))

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
)

# ----------------------------------------------------------- Defining Logger file to track performance --------------------------------------------- #
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

############################################## - END - SAME CODE FOR ALL - #####################################################################

## --------------------------------------------------- SET CNN Model -------------------------------------------------------- #

# Define the AlexNet model
def alexnet(input_shape=(224, 224, 3)):
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # Flatten and Fully Connected Layers
    model.add(Flatten())

    # 1st Fully Connected Layer
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # 2nd Fully Connected Layer
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))
    
    return model

model = alexnet(input_shape=(224, 224, 3))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])



############################################################### - START - SAME CODE AGAIN - ####################################################################

# ------------------------------------------------------- Defining Callbacks ------------------------------------------------------------------------ # 
# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=f'./{output_dir}/{model}_{method}_best_weights.tf', monitor='val_loss', save_best_only=True, mode='min', save_format="tf")
csv_logger = CSVLogger(f'./{output_dir}/{model}_{method}_training_log.csv')

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=epochs,
    callbacks=[
        #early_stopping,
        checkpoint,
        csv_logger
    ]
)

# Load the best weights
model.load_weights(f'./{output_dir}/{model}_{method}_best_weights.tf')

# Evaluate the model
val_loss, val_acc = model.evaluate(val_generator)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_acc}')

############################################################### - END - SAME CODE AGAIN - ####################################################################
