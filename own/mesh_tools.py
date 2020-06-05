# Wojciech Golaszewski

from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import trimesh
from numpy import sin, cos, pi
import numpy as np
import pyrender
import pyglet
import sys
import os

# colors
cs = {
    "blue_bright": [0.0, 0.3, 1.0],
    "orange": [1.0, 0.3, 0.0],
    "red": [1.0, 0.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "purple": [0.6, 0.0, 0.8],
}


def color_verts(self, color):
    colors = np.ones((self.vertices.shape[0], 3)) * color
    self.visual.vertex_colors = colors
    return self


trimesh.Trimesh.color_verts = color_verts


def rotate_in_yaxis(self, angle_rad):
    a = angle_rad
    s = sin(a)
    c = cos(a)
    y_rot = [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]
    self.apply_transform(y_rot)
    return self


trimesh.Trimesh.rotate_in_yaxis = rotate_in_yaxis


def apply_rotation(self, R):
    T = np.zeros((4, 4))
    T[:3, :3] = R
    return self.apply_transform(T)


trimesh.Trimesh.apply_rotation = apply_rotation


def get_line_as_cuboid(scale=1, thickness=1):
    mesh_x = trimesh.creation.box()
    mesh_x.vertices = (
        (mesh_x.vertices + [0.5, 0, 0]) * [10 * thininess, 1, 1] * scale / thininess
    )
    mesh_x.color_verts([1, 0, 0])


def get_axis_trimesh(scale=1, thickness=1):

    scale = scale / 10
    thininess = 1 / thickness * 5

    mesh_x = trimesh.creation.box()
    mesh_x.vertices = (
        (mesh_x.vertices + [0.5, 0, 0]) * [10 * thininess, 1, 1] * scale / thininess
    )
    mesh_x.color_verts([1, 0, 0])

    mesh_y = trimesh.creation.box()
    mesh_y.vertices = (
        (mesh_y.vertices + [0, 0.5, 0]) * [1, 10 * thininess, 1] * scale / thininess
    )
    mesh_y.color_verts([0, 1, 0])

    mesh_z = trimesh.creation.box()
    mesh_z.vertices = (
        (mesh_z.vertices + [0, 0, 0.5]) * [1, 1, 10 * thininess] * scale / thininess
    )
    mesh_z.color_verts([0, 0, 1])
    # run_gui_tmesh([mesh_x, mesh_y, mesh_z])

    axis_mesh = trimesh.Trimesh(
        vertices=np.concatenate([mesh_x.vertices, mesh_y.vertices, mesh_z.vertices]),
        faces=np.concatenate([mesh_x.faces, mesh_y.faces + 8, mesh_z.faces + 16]),
        vertex_colors=np.concatenate(
            [mesh.visual.vertex_colors for mesh in [mesh_x, mesh_y, mesh_z]]
        ),
    )

    return axis_mesh


def generateSceneWithPyrenderMeshes(pyr_meshes, ambient_white_light=0.5):
    scene = pyrender.Scene(ambient_light=ambient_white_light * np.ones(3))

    for pyr_mesh in pyr_meshes:
        scene.add(pyr_mesh)

    return scene


def generateSceneWithTmeshes(tmeshes, ambient_white_light=0.5):

    pyr_meshes = [pyrender.Mesh.from_trimesh(tmesh) for tmesh in tmeshes]
    return generateSceneWithPyrenderMeshes(pyr_meshes, ambient_white_light)


def run_gui_tmesh(tmeshes):
    scene = generateSceneWithTmeshes(tmeshes)
    v = pyrender.Viewer(scene, use_raymond_lighting=True)
    del v


def run_gui_pyrmesh(pyr_meshes, point_size=10):
    scene = generateSceneWithPyrenderMeshes(pyr_meshes)
    v = pyrender.Viewer(
        scene, use_raymond_lighting=True, point_size=point_size, refresh_rate=60,
    )
    del v


def get_offline_render(tmeshes):
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0.1], [0, 0, 1, 0.3], [0, 0, 0, 1]])

    scene = generateSceneWithTmeshes(tmeshes, ambient_white_light=1)
    scene.add(camera, pose=camera_pose)

    r = pyrender.OffscreenRenderer(viewport_width=400, viewport_height=400)
    color, depth = r.render(scene)
    r.delete()

    return color


def run_offline_tmesh(tmeshes):

    color = get_offline_render(tmeshes)

    plt.figure(dpi=100)
    plt.imshow(color)
    plt.axis("off")
    plt.show()


def get_noisy_mesh(tmesh, constant):
    tm_noisy = tmesh.copy()

    tm_bounds_mean_range = np.mean(abs(tm_noisy.bounds[0, :] - tm_noisy.bounds[1, :]))
    noise_multiplier = constant * tm_bounds_mean_range

    tm_noisy.vertices += np.random.randn(*tm_noisy.vertices.shape) * noise_multiplier

    return tm_noisy
