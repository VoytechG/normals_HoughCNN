import sys, os

import pyglet

pyglet.options["shadow_window"] = False

import pyrender
import numpy as np
import trimesh
import random
import time

from scipy.sparse.linalg import eigs
from mpl_toolkits import mplot3d
from sklearn import preprocessing

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def evaluate_normals(original, estimate):
    maxdeg = 30
    # Calculate the angle difference between the two normals
    size = original.shape[0]
    evaluation = np.zeros(size)

    for i in range(size):
        org = np.array(original[i, 3:])
        est = np.array(estimate[i, 3:])
        # dot = np.dot(original[i, 3:], estimate[i, 3:])
        # det = np.linalg.norm(original[i, 3:]) * np.linalg.norm(estimate[i, 3:])
        # test = dot / det

        # # Ensures that the cos value is valid
        # if test > 1.0:
        #     test = 1.0
        # if test < -1.0:
        #     test = -1.0
        # evaluation[i] = np.degrees(np.arccos(test))

        error_angle = min(angle_between(org, est), angle_between(org, -est))
        error_angle = np.degrees(error_angle)

        # We set any angle deviation beyond the range to the maximum
        if error_angle > maxdeg:
            error_angle = maxdeg

        evaluation[i] = error_angle

    # Normalises the evaluation data
    evaluation = evaluation / maxdeg

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    # levels = np.linspace(0.0, 20.0, 7)
    colours = ax.scatter(
        original[:, 0], original[:, 1], original[:, 2], c=evaluation, cmap="jet"
    )
    ticks = np.arange(0, maxdeg + 1, maxdeg / 5)

    cbar = fig.colorbar(colours, ticks=ticks / maxdeg)
    # cbar = fig.colorbar(colours, ticks=ticks)
    cbar.ax.set_yticklabels([str(int(tick)) for tick in ticks])
    # cbar.ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "0.95", "1.0"])

    # cbar = fig.colorbar(colours)


def load_xyz(path):
    lines = open(path).read().split("\n")[:-1]
    xyz = np.array([[float(x) for x in line.split(" ")[:6]] for line in lines])
    return xyz


# gt = load_xyz("inputs/rect_small.xyz")
# pr = load_xyz("outputs/rect_small_better_normals.xyz")

# evaluate_normals(gt, pr)
