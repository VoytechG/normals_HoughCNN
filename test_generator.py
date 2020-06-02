import os
import python.lib.python.NormalEstimatorHoughCNN as Estimator

estimator = Estimator.NormalEstimatorHoughCNN()

batch_size = 32

nbr, batch, batch_targets = estimator.generate_training_accum_random_corner(batch_size)

dataset = {"nbr": nbr, "batch": batch, "batch_targets": batch_targets}

print("  saving")
os.makedirs(dataset_directory, exist_ok=True)
pickle.dump(dataset, open(os.path.join(dataset_directory, "dataset.p"), "wb"))
print("-->done")
