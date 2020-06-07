from tensorflow import keras
from tensorflow.keras import layers
import pickle
from matplotlib.pyplot import imshow
import os
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

# from main import get_batch
from hough_estimator import HoughEstimator as HE

filename = "cow"
input_filename = "inputs/" + filename + ".xyz"
output_filename = "outputs/" + filename + "_better_normals_2.xyz"
predictions_filename = "outputs/" + filename + "_predictions_2.npy"

model_file_name = "model_c3_15.h5"

dataset = pickle.load(open(os.path.join(dataset_directory, dataset_filename), "rb"))
mean = dataset["mean"]


def restore_normal(normal, R_inv):
    normal = np.array([normal[0], normal[1], 0])
    norm = np.linalg.norm(normal)
    if norm >= 1:
        normal = normal / norm
    else:
        normal[2] = (1 - norm ** 2) ** 0.5
    normal = R_inv @ normal
    normal = normal / np.linalg.norm(normal)
    return normal


def predict(point_cloud):
    print(f"Generating accumulators for {len(point_cloud)} points")
    he = HE(number_of_channels=3, point_cloud=point_cloud)
    inputs, inverse_rotation_matrices = he.generate_accums_for_point_cloud()

    inputs -= mean

    print("Running CNN... ")
    model = keras.models.load_model(keras_model_save_path + model_file_name)
    predictions = model.predict(inputs)

    with open(predictions_filename, "wb") as f:
        np.save(f, predictions)

    predicted_normals = [
        restore_normal(normal, R_inv[0])
        for normal, R_inv in zip(predictions, inverse_rotation_matrices)
    ]

    predicted_xyz = np.concatenate([point_cloud, predicted_normals], 1)
    np.savetxt(output_filename, predicted_xyz)

    return predicted_normals


point_cloud = HE.load_point_cloud(input_filename)
predict(point_cloud)
