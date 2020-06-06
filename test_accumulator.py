import pickle
from  matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import os
import numpy as np

dataset_directory = "dataset/"    
dataset_filename = "datasetk3.p"
# dataset_directory = "generated_inputs/"
# dataset_filename = "small_only_3d_pca.p" 

dataset = pickle.load( open( os.path.join(dataset_directory, dataset_filename), "rb" ) )

X = dataset['input']
X.shape
# M = dataset['mean']
# imshow(M[2])

plt.ion()
plt.show()

for i in range(100):
    plt.imshow(X[i, 1])
    plt.draw()
    plt.pause(0.01)
    input("Press [enter] to continue.")