Ks = [0.5, 1, 2]

batch_size = 80
batches_to_generate = 40000 // batch_size
# batches_to_generate = 10

channels = len(Ks)
epochs = 40

MIN_ANGLE = 80
MAX_ANGLE = 160

MIN_NOISE_FACTOR = 0
MAX_NOISE_FACTOR = 1.5

processes_to_use = 4

keras_model_save_path = "keras_models/"

dataset_directory = "generated_inputs"
dataset_filename = "dataset_w3.p"
