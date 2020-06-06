import sys, os

import pyglet
import pyrender
import numpy as np
import trimesh
import random
import time

from scipy.sparse.linalg import eigs
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation as R


def generate_point_cloud(angle, noise_factor=0):
    # Control variables

    # Scale changes size
    scale = 8
    # Degrees changes angles of the walls
    if abs(angle - 90) < 0.00001:
        angle = 90.1
    degrees = angle - 90
    # Rotator changes the point cloud's entire rotation
    rotator = 90

    # NoiseSigma changes the intensity of the noise applied
    sigma = 0.01
    mu = 0

    # Ensures we get 5000 points total
    # maxiter1 is for initial wall generation
    maxiter1 = 1249
    # maxiter2 is for rotation merging between Wall 1 and Wall 2 (the two vertical walls)
    maxiter2 = 625

    # Sets vertices as blank so that we can refer to it outside of functions
    vertices = np.zeros([5000, 6])

    # Generates random points in the point cloud, starting with simple 90-degree planes
    # First 3 columns are x-y-z coordinates
    # Last 3 columns are x-y-z normal vectors

    wall1 = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    wall2 = np.array([[0.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    wall3 = np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])

    for i in range(maxiter1):

        r1 = random.uniform(0.0, 1.0)
        r2 = random.uniform(0.0, 1.0)
        r3 = random.uniform(0.0, 1.0)
        r4 = random.uniform(0.0, 1.0)
        r5 = random.uniform(0.0, 1.0)
        r6 = random.uniform(0.0, 1.0)

        wall1 = np.append(wall1, [[r1, r2, 0.0, 0.0, 0.0, 1.0]], axis=0)
        wall2 = np.append(wall2, [[0.0, r3, r4, 1.0, 0.0, 0.0]], axis=0)
        wall3 = np.append(wall3, [[r5, 0.0, r6, 0.0, 1.0, 0.0]], axis=0)

    rotate1 = Rotation.from_euler("x", -degrees, degrees=True)
    rotate2 = Rotation.from_euler("z", degrees, degrees=True)

    # Calculates the point where the two walls would meet when rotated
    minval = np.dot(rotate1.as_matrix(), np.array(wall1[0, :3]).T)[2]

    temp1 = np.zeros([1, 6])
    temp2 = np.zeros([1, 6])

    # Adds more points to the pointcloud to merge the walls together
    for i in range(maxiter2):

        r1 = random.uniform(minval, 0.0)
        r4 = random.uniform(minval, 0.0)

        min1 = r1 / minval
        min2 = r4 / minval
        r2 = random.uniform(min1, 1.0)
        r3 = random.uniform(min2, 1.0)

        temp1 = np.append(temp1, [[r1, r2, 0.0, 0.0, 0.0, 1.0,]], axis=0)
        temp2 = np.append(temp2, [[0.0, r3, r4, 1.0, 0.0, 0.0,]], axis=0)

    # Rotates the walls to join together
    walljoint1 = np.dot(rotate1.as_matrix(), np.array(temp1[1:, :3]).T).T
    walljoint2 = np.dot(rotate2.as_matrix(), np.array(temp2[1:, :3]).T).T
    walljointnormals1 = np.dot(rotate1.as_matrix(), np.array(temp1[1:, 3:]).T).T
    walljointnormals2 = np.dot(rotate2.as_matrix(), np.array(temp2[1:, 3:]).T).T

    walljoint1 = np.append(walljoint1, walljointnormals1, axis=1)
    walljoint2 = np.append(walljoint2, walljointnormals2, axis=1)

    # --------- #

    # Merges all of the points together for a single point cloud file

    rotatedWall1 = np.dot(rotate1.as_matrix(), np.array(wall1[:, :3]).T).T
    rotatedWall2 = np.dot(rotate2.as_matrix(), np.array(wall2[:, :3]).T).T

    rotatedNormals1 = np.dot(rotate1.as_matrix(), np.array(wall1[:, 3:]).T).T
    rotatedNormals2 = np.dot(rotate2.as_matrix(), np.array(wall2[:, 3:]).T).T

    rotatedWall1 = np.append(rotatedWall1, rotatedNormals1, axis=1)
    rotatedWall2 = np.append(rotatedWall2, rotatedNormals2, axis=1)

    fullwall1 = np.append(rotatedWall1, walljoint1, axis=0)
    fullwall2 = np.append(rotatedWall2, walljoint2, axis=0)

    fullwall3 = np.array(wall3)

    fullwall1[:, :3] = fullwall1[:, :3] * scale
    fullwall2[:, :3] = fullwall2[:, :3] * scale
    fullwall3[:, :3] = wall3[:, :3] * scale

    vertices = np.append(fullwall1, fullwall2, axis=0)
    vertices = np.append(vertices, fullwall3, axis=0)

    # Finally, rotates the point cloud to its new initial orientation
    if rotator != 0:
        rotatePoints = Rotation.from_euler("x", rotator, degrees=True)
        vertices[:, :3] = np.dot(
            rotatePoints.as_matrix(), np.array(vertices[:, :3]).T
        ).T
        vertices[:, 3:] = np.dot(
            rotatePoints.as_matrix(), np.array(vertices[:, 3:]).T
        ).T

        fullwall1[:, :3] = np.dot(
            rotatePoints.as_matrix(), np.array(fullwall1[:, :3]).T
        ).T
        fullwall1[:, 3:] = np.dot(
            rotatePoints.as_matrix(), np.array(fullwall1[:, 3:]).T
        ).T
        fullwall2[:, :3] = np.dot(
            rotatePoints.as_matrix(), np.array(fullwall2[:, :3]).T
        ).T
        fullwall2[:, 3:] = np.dot(
            rotatePoints.as_matrix(), np.array(fullwall2[:, 3:]).T
        ).T
        fullwall3[:, :3] = np.dot(
            rotatePoints.as_matrix(), np.array(fullwall3[:, :3]).T
        ).T
        fullwall3[:, 3:] = np.dot(
            rotatePoints.as_matrix(), np.array(fullwall3[:, 3:]).T
        ).T

    point_cloud, normals = vertices[:, :3], vertices[:, 3:]

    kdtree = KDTree(point_cloud)
    distances, _ = kdtree.query(point_cloud, 2)
    mean_dst = np.mean(distances[:, 1])

    point_cloud += np.random.randn(*point_cloud.shape) * mean_dst * noise_factor

    R = Rotation.random().as_matrix()

    point_cloud = (R @ point_cloud.T).T
    normals = (R @ normals.T).T

    return point_cloud, normals


def save_file(fliename, point_cloud, normals):
    np.savetxt(fliename, np.concatenate([point_cloud, normals], 1))
