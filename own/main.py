import time
import numpy as np
import os
import pickle

import hough_estimator as he
import multiprocessing
from config import Ks

dataset_directory = "generated_inputs"
dataset_filename = "small_only_3d_pca.p"


def get_batch(index):
    print(f"Working on {index}")
    index *= 256
    HE = he.HoughEstimator(
        point_cloud_path=f"../3dmodels/model_{index}.xyz", K_multipliers=Ks
    )
    HE.T = 1000
    accumulators, _ = HE.generate_accums_for_point_cloud(number_of_points=64)

    return accumulators


if __name__ == "__main__":

    pool = multiprocessing.Pool(7)
    start_time = time.time()
    accumulators = np.concatenate(pool.map(get_batch, list(range(10))))
    print("--- %s seconds ---" % (time.time() - start_time))

    dataset = {"inputs": accumulators, "mean": np.mean(accumulators, 0)}
    print("saving")
    os.makedirs(dataset_directory, exist_ok=True)
    pickle.dump(dataset, open(os.path.join(dataset_directory, dataset_filename), "wb"))
