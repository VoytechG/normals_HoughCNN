from tensorflow import keras
from tensorflow.keras import layers
import pickle
from matplotlib.pyplot import imshow
import os
from tqdm import *

import numpy as np
from config import (
    channels,
    batch_size,
    keras_model_save_path,
    Ks,
    dataset_directory,
    dataset_filename,
)
from sklearn.model_selection import train_test_split
from cython import NormalEstimatorHoughCNN as Estimator

filename = "rect_small.xyz"
input_filename = f"inputs/{filename}"
output_filename = f"outputs/{filename}"

estimator = Estimator.NormalEstimatorHoughCNN()
estimator.loadXYZ(input_filename)
estimator.set_Ks(np.array(Ks))
estimator.initialize()

# mean = np.load(f"../trained_model/mean.npz")["arr_0"]
mean = pickle.load(open(os.path.join(dataset_directory, dataset_filename), "rb"))[
    "mean"
]

# model_file_name = "model_1.h5"
model_file_name = "model_c3_1.h5"
model = keras.models.load_model(keras_model_save_path + model_file_name)

batch_size = 512
# for pt_id in range(0, estimator.size(), batch_size):
#     batch = estimator.get_batch(pt_id, batch_size) - mean[None, :, :, :]
#     estimations = model.predict(batch)

for pt_id in tqdm(range(0, estimator.size(), batch_size)):
    bs = batch_size
    batch = estimator.get_batch(pt_id, bs) - mean
    # batch_th = torch.Tensor(batch)
    # estimations = model.predict(batch_th)
    # estimations = estimations.cpu().data.numpy()
    estimations = model.predict(batch_th)
    # estimator.set_batch(pt_id, bs, estimations.astype(np.float64))
    estimator.set_batch(pt_id, bs, np.array(estimations, dtype="float64"))
