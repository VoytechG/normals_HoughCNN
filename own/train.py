from tensorflow import keras
from tensorflow.keras import layers
import pickle
from matplotlib.pyplot import imshow
import os
import numpy as np
from config import (
    channels,
    keras_model_save_path,
    epochs,
    dataset_directory,
    dataset_filename,
)
from sklearn.model_selection import train_test_split


def create_model():
    conv_layer_params = {
        "activation": "relu",
        "data_format": "channels_first",
        "padding": "valid",
    }
    max_pool_layer_params = {
        "data_format": "channels_first",
        "padding": "valid",
    }
    dense_layer_params = {"activation": "relu"}

    model = keras.Sequential()
    sequence = [
        layers.Conv2D(50, 3, input_shape=(channels, 33, 33), **conv_layer_params),
        layers.BatchNormalization(),
        layers.Conv2D(50, 3, **conv_layer_params),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, **max_pool_layer_params),
        # layers.Conv2D(50, 3, **conv_layer_params),
        # layers.BatchNormalization(),
        layers.Conv2D(96, 3, **conv_layer_params),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, **max_pool_layer_params),
        layers.Flatten(),
        layers.Dense(2048, **dense_layer_params),
        layers.Dropout(0.5),
        layers.Dense(1024, **dense_layer_params),
        layers.Dropout(0.5),
        layers.Dense(512, **dense_layer_params),
        layers.Dense(2),
    ]

    for lay in sequence:
        model.add(lay)

    return model


def load_data():
    dataset = pickle.load(open(os.path.join(dataset_directory, dataset_filename), "rb"))
    return dataset


data = load_data()
X = data["inputs"] - data["mean"]
y = data["targets"]

model = create_model()
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-5), loss=keras.losses.MSE,
)

model.summary()
model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=128, shuffle=True)

model.save(keras_model_save_path + "model_c3_15.h5")
