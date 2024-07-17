import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import os,logging,csv

############################################################# - START - SAME CODE FOR ALL - ########################################################################

## --------------------------------------------------------------------- Defining Edge Detection Method -------------------------------------------------------- #
epochs = 100
model='inceptionv3'
method='roberts'

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

