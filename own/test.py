from tensorflow import keras
from tensorflow.keras import layers
import pickle
from matplotlib.pyplot import imshow
import os
import numpy as np
from config import channels, batch_size, keras_model_save_path, Ks
from sklearn.model_selection import train_test_split
from ..python.lib.python import NormalEstimatorHoughCNN as Estimator

filename = "rect.xyz"
input_filename = f"../inputs/{filename}"
output_filename = f"../outputs/{filename}"

estimator = Estimator.NormalEstimatorHoughCNN()
estimator.loadXYZ(input_filename)
estimator.set_Ks(Ks)
estimator.initialize()

mean = np.load(f"../trained_model/mean.npz")["arr_0"]

model_file_name = "model_1.h5"
model = keras.models.load_model(keras_model_save_path + model_file_name)

batch_size = 512
for pt_id in range(0, estimator.size(), batch_size):
    batch = estimator.get_batch(pt_id, batch_size) - mean[None, :, :, :]
    estimations = model.predict(batch)
