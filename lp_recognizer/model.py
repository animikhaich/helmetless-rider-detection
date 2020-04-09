# Imports
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Nadam, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tensorflow.keras.applications.nasnet import NASNetLarge, NASNetMobile
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet import ResNet50, ResNet152
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet152V2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, MaxPool2D, Dropout, BatchNormalization, Input, Activation, LeakyReLU, add
from tensorflow.keras.models import Model, clone_model
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import os
import gc
import datetime
import pickle
import json

import tensorflow as tf

# Allow GPU Memory Growth
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

# Enable XLA
tf.config.optimizer.set_jit(True)

# Keras imports

# Define Constants
HEIGHT = 137
WIDTH = 236
SIZE = 224
BATCH_SIZE = 64

base_name = "CustomResNet_5_" + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
TB_LOG_DIR = "logs/" + base_name
WEIGHTS_PATH = f"saved_model/{base_name}.h5"
CONFIG_PATH = f"saved_model/{base_name}.cfg"

# Set Mixed Precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Make image dumping directory
train_dir_path = './data/images/train'
test_dir_path = './data/images/test'

if not os.path.isdir(train_dir_path):
    os.makedirs(train_dir_path)

if not os.path.isdir(test_dir_path):
    os.makedirs(test_dir_path)

model_dir = os.path.split(WEIGHTS_PATH)[0]
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

# Read train labels
train_label_path = "./data/train.csv"
df = pd.read_csv(train_label_path)

# Dropping the last column (not required for training)
df.drop(['grapheme'], axis=1, inplace=True)

# Separating the data from the labels and one hot encoding the labels
y_root = pd.get_dummies(df['grapheme_root'])
y_vowel = pd.get_dummies(df['vowel_diacritic'])
y_consonant = pd.get_dummies(df['consonant_diacritic'])
x_name = df['image_id']

# Train Test Split
x_train_name, x_val_name, y_root_train, y_root_val, y_vowel_train, y_vowel_val, y_consonant_tarin, y_consonant_val = train_test_split(
    x_name, y_root, y_vowel, y_consonant, test_size=0.2, random_state=3)

# Create the Data Generator


def data_generator(x, y_root, y_vowel, y_consonant, batch_size=16, saved_img_path='./data/images/train', image_shape=(299, 299)):
    assert len(x) == len(y_root) == len(y_vowel) == len(
        y_consonant), 'Lengths of all inputs should be same'

    num_splits = round(len(x) // batch_size) + 1

    x_splits = np.array_split(x, num_splits)

    y_root_splits = np.array_split(y_root, num_splits)
    y_vowel_splits = np.array_split(y_vowel, num_splits)
    y_consonant_splits = np.array_split(y_consonant, num_splits)

    i = 0

    while True:
        xs = list()

        x_batch = x_splits[i].values
        y_root_batch = y_root_splits[i].values
        y_vowel_batch = y_vowel_splits[i].values
        y_consonant_batch = y_consonant_splits[i].values

        i += 1
        if i > num_splits-1:
            i = 0

        for x_ in x_batch:
            path = os.path.join(saved_img_path, f"{x_}.jpg")
            image = cv2.resize(cv2.imread(path, 0),
                               image_shape, cv2.INTER_AREA)/255
            xs.append(np.expand_dims(image, axis=2))

        yield np.array(xs), [y_root_batch, y_vowel_batch, y_consonant_batch]


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

# Model
# inputs = Input(shape = (SIZE, SIZE, 3))
# model = Conv2D(filters=3, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(SIZE, SIZE, 1))(inputs)
# model = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(model)
# base = ResNet50V2(include_top=False, weights=None, input_shape=(SIZE, SIZE, 3))
# model = Flatten()(base.output)
# model = Dense(2048, activation="linear")(model)
# model = Dropout(rate=0.5)(model)
# dense = Dense(1024, activation="relu")(model)
# model = Dropout(rate=0.5)(model)
# head_root = Dense(168, activation='softmax', name="root")(dense)
# head_vowel = Dense(11, activation='softmax', name="vowel")(dense)
# head_consonant = Dense(7, activation='softmax', name="consonant")(dense)
# model = Model(inputs=base.inputs, outputs=[head_root, head_vowel, head_consonant])

"""
inputs = Input(shape = (SIZE, SIZE, 1))
model = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
model = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Dropout(0.25)(model)
model = Conv2D(64, (3, 3), activation='relu', padding='same')(model)
model = Conv2D(64, (3, 3), activation='relu', padding='same')(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Dropout(0.25)(model)
model = Conv2D(128, (3, 3), activation='relu', padding='same')(model)
model = Conv2D(128, (3, 3), activation='relu', padding='same')(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Dropout(0.25)(model)
model = Flatten()(model)
model = Dense(1024, activation='relu')(model)
model = Dropout(0.5)(model)
model = Dense(512, activation='relu')(model)
model = Dropout(0.5)(model)
head_root = Dense(168, activation='softmax', name="root")(model)
head_vowel = Dense(11, activation='softmax', name="vowel")(model)
head_consonant = Dense(7, activation='softmax', name="consonant")(model)
model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])
"""


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
    in_layer = Dense(1024)(in_layer)
    in_layer = BatchNormalization(momentum=0.2)(in_layer)
    in_layer = LeakyReLU()(in_layer)
    in_layer = Dropout(0.5)(in_layer)

    in_layer = Dense(512)(in_layer)
    in_layer = BatchNormalization(momentum=0.2)(in_layer)
    in_layer = LeakyReLU()(in_layer)
    in_layer = Dropout(0.5)(in_layer)

    in_layer = Dense(256)(in_layer)
    in_layer = BatchNormalization(momentum=0.2)(in_layer)
    in_layer = LeakyReLU()(in_layer)
    in_layer = Dropout(0.5)(in_layer)

    if num_layers == 4:
        in_layer = Dense(128)(in_layer)
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

pre_root = DenseInterNet(model, num_layers=3)
head_root = Dense(168)(pre_root)
head_root = Activation('softmax', dtype='float32', name='root')(head_root)

pre_vowel = DenseInterNet(model, num_layers=4)
head_vowel = Dense(11)(pre_vowel)
head_vowel = Activation('softmax', dtype='float32', name='vowel')(head_vowel)

pre_consonant = DenseInterNet(model, num_layers=4)
head_consonant = Dense(7)(pre_consonant)
head_consonant = Activation(
    'softmax', dtype='float32', name='consonant')(head_consonant)

model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])

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

# Create Generators
train_gen = data_generator(x_train_name, y_root_train, y_vowel_train, y_consonant_tarin,
                           batch_size=BATCH_SIZE, image_shape=(SIZE, SIZE), saved_img_path='/home/ani/Documents/temp')
val_gen = data_generator(x_val_name, y_root_val, y_vowel_val, y_consonant_val,
                         batch_size=BATCH_SIZE, image_shape=(SIZE, SIZE), saved_img_path='/home/ani/Documents/temp')

# Train the model
# Fit the model
history = model.fit_generator(train_gen, epochs=500, validation_data=val_gen,
                              steps_per_epoch=len(
                                  x_train_name)//BATCH_SIZE + 1,
                              validation_steps=len(x_val_name)//BATCH_SIZE + 1,
                              callbacks=[tensorboard_callback,
                                         checkpoint_callback,
                                         early_stopping_callback,
                                         reduce_lr_callback])

