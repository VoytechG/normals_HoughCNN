import time
import hough_estimator as he
import multiprocessing

Ks = [0.5, 1, 2]


def get_batch(index):
    print(f"Working on {index}")
    index *= 256
    HE = he.HoughEstimator(
        point_cloud_path=f"../3dmodels/model_{index}.xyz", K_multipliers=Ks
    )
    # HE.T = 300
    HE.generate_accums_for_point_cloud(number_of_points=64)


paths = ["model_0.xyz", "model_256.xyz", "model_512.xyz"]

if __name__ == "__main__":

    pool = multiprocessing.Pool(7)
    start_time = time.time()
    results = list(map(get_batch, list(range(10))))
    print("--- %s seconds ---" % (time.time() - start_time))
