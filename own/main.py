import time
import numpy as np
import os
import pickle

from hough_estimator import HoughEstimator as HE
import multiprocessing
from config import (
    Ks,
    batch_size,
    MIN_ANGLE,
    MAX_ANGLE,
    MIN_NOISE_FACTOR,
    MAX_NOISE_FACTOR,
    batches_to_generate,
    dataset_directory,
    dataset_filename,
    processes_to_use,
)
from mesh_generator import generate_point_cloud

parallel_computing = True

visualise_hypothesis = False
visualise_accumulator = False
visualise_valid_points = False


def get_batch(index):

    start_time_batch = time.time()

    angle = np.random.rand() * (MAX_ANGLE - MIN_ANGLE) + MIN_ANGLE
    noise_factor = (
        np.random.rand() * (MAX_NOISE_FACTOR - MIN_NOISE_FACTOR) + MIN_NOISE_FACTOR
    )

    point_cloud, normals = generate_point_cloud(angle, noise_factor)

    # print(f"Working on {index}")
    # index *= 256
    # point_cloud = HE.load_point_cloud(f"../3dmodels/model_{index}.xyz")
    # normals = None

    he = HE(number_of_channels=3, point_cloud=point_cloud, ground_truth_normals=normals)
    he.VISUALISE_HYPOTHESIS = visualise_hypothesis
    he.VISUALISE_ACCUMULATOR = visualise_accumulator
    he.VISUALISE_VALID_POINTS = visualise_valid_points

    (
        accumulators,
        inverse_rotation_matrices,
        target_normals,
    ) = he.generate_accums_for_point_cloud(batch_size=batch_size)

    batch_duration = time.time() - start_time_batch
    print(f"finished {index+1}/{batches_to_generate} --- {batch_duration:.2f}s ---")

    return accumulators, inverse_rotation_matrices, target_normals


if __name__ == "__main__":

    start_time = time.time()

    print(
        f"Generating {batches_to_generate} batches with {batch_size} samples each "
        + f"({batches_to_generate * batch_size} total)"
    )

    if parallel_computing:
        pool = multiprocessing.Pool(processes_to_use)
        batches = list(pool.map(get_batch, list(range(batches_to_generate))))
    else:
        batches = list(map(get_batch, list(range(batches_to_generate))))

    accumulators = np.concatenate([batch[0] for batch in batches])
    inv_rot_matrices = np.concatenate([batch[1] for batch in batches])
    target_normals = np.concatenate([batch[2] for batch in batches])

    print("--- %s seconds ---" % (time.time() - start_time))

    dataset = {
        "inputs": accumulators,
        "mean": np.mean(accumulators, 0),
        "inv_rot_matrices": inv_rot_matrices,
        "targets": target_normals,
    }
    print("saving")
    os.makedirs(dataset_directory, exist_ok=True)
    pickle.dump(dataset, open(os.path.join(dataset_directory, dataset_filename), "wb"))
