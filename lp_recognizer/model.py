# Imports
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, MaxPool2D, Dropout, BatchNormalization, Input, Activation, LeakyReLU, add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Nadam, Adagrad
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import pickle
import json
import cv2
import os
import gc


# Allow GPU Memory Growth
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

# Enable XLA
tf.config.optimizer.set_jit(True)

# Define Constants
SIZE = 100
BATCH_SIZE = 128

base_name = "CustomResNet_" + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
TB_LOG_DIR = "tf_logs/" + base_name
WEIGHTS_PATH = f"weights/{base_name}.h5"
CONFIG_PATH = f"weights/{base_name}.cfg"

# Set Mixed Precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Make image dumping directory
train_dir_path = '/home/ani/Documents/ActiveTrainingData/OCR/splitted/train'
test_dir_path = '/home/ani/Documents/ActiveTrainingData/OCR/splitted/val'


train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rescale=1/255.,
    brightness_range=[0.4, 1.0],
    zoom_range=[0.7, 1.3]
)

val_datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    train_dir_path,
    target_size=(SIZE, SIZE),
    color_mode='grayscale',
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=10
)

val_gen = val_datagen.flow_from_directory(
    test_dir_path,
    target_size=(SIZE, SIZE),
    color_mode='grayscale',
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=10
)

train_steps_per_epoch = int(train_gen.n / train_gen.batch_size)
val_steps_per_epoch = int(val_gen.n / val_gen.batch_size)

# Callbacks
tensorboard_callback = TensorBoard(
    log_dir=TB_LOG_DIR,
    histogram_freq=1,
    write_images=False,
    update_freq='batch'
)
checkpoint_callback = ModelCheckpoint(
    filepath=WEIGHTS_PATH, monitor='val_loss', verbose=0, save_weights_only=True, save_best_only=True)
early_stopping_callback = EarlyStopping(
    monitor='val_loss', patience=6, verbose=1)
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, min_lr=1e-6, patience=2, verbose=1)


def ResidualBlock(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = Conv2D(nb_channels, kernel_size=(3, 3),
               strides=_strides, padding='same')(y)
    y = BatchNormalization(momentum=0.2)(y)
    y = LeakyReLU()(y)

    y = Conv2D(nb_channels, kernel_size=(3, 3),
               strides=_strides, padding='same')(y)
    y = BatchNormalization(momentum=0.2)(y)
    y = LeakyReLU()(y)

    y = Conv2D(nb_channels, kernel_size=(3, 3),
               strides=(1, 1), padding='same')(y)
    y = BatchNormalization(momentum=0.2)(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = Conv2D(nb_channels, kernel_size=(
            1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = BatchNormalization(momentum=0.2)(shortcut)

    y = add([shortcut, y])
    y = LeakyReLU()(y)

    return y


def DenseInterNet(in_layer, num_layers=4):
    in_layer = Dense(256)(in_layer)
    in_layer = BatchNormalization(momentum=0.2)(in_layer)
    in_layer = LeakyReLU()(in_layer)
    in_layer = Dropout(0.5)(in_layer)

    if num_layers == 4:
        in_layer = Dense(64)(in_layer)
        in_layer = BatchNormalization(momentum=0.2)(in_layer)
        in_layer = LeakyReLU()(in_layer)
        in_layer = Dropout(0.5)(in_layer)

    return in_layer


inputs = Input(shape=(SIZE, SIZE, 1))
model = ResidualBlock(y=inputs, nb_channels=32, _project_shortcut=True)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Dropout(0.5)(model)

model = ResidualBlock(y=model, nb_channels=64, _project_shortcut=True)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Dropout(0.5)(model)

model = ResidualBlock(y=model, nb_channels=128, _project_shortcut=True)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Dropout(0.5)(model)

model = ResidualBlock(y=model, nb_channels=256, _project_shortcut=True)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Dropout(0.5)(model)

model = Flatten()(model)


model = DenseInterNet(model, num_layers=4)
model = Dense(train_gen.num_classes)(model)
model = Activation('softmax', dtype='float32', name='lp_ocr_out')(model)

model = Model(inputs=inputs, outputs=model)

# Model Summary
model.summary()

# Compile Model
opt = Adam(learning_rate=0.001)
# opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt) # Enable Tensorflow Mixed Precision
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[
              'accuracy', Precision(), Recall(), 'mean_squared_error'])

# Get the model Config
cfg = model.get_config()
with open(CONFIG_PATH, 'w') as f:
    json.dump(cfg, f, indent=4)


# Train the model
# Fit the model
history = model.fit_generator(train_gen, epochs=500, validation_data=val_gen,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_steps=val_steps_per_epoch,
                              callbacks=[tensorboard_callback,
                                         checkpoint_callback,
                                         early_stopping_callback,
                                         reduce_lr_callback])
