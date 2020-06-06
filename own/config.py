Ks = [0.5, 1, 2]

batch_size = 128
batches_to_generate = 60000 // batch_size
# batches_to_generate = 10

channels = len(Ks)
epochs = 40

MIN_ANGLE = 80
MAX_ANGLE = 160

MIN_NOISE_FACTOR = 0
MAX_NOISE_FACTOR = 2

processes_to_use = 7

keras_model_save_path = "keras_models/"

dataset_directory = "generated_inputs"
dataset_filename = "small_1.p"
