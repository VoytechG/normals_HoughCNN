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

RES_PATH = './'
LIB_PATH = './python_lib'

if not os.path.exists(RES_PATH):
    print('cannot find resources, please update RES_PATH')
    exit(1)
else:
    print('found resources')

# append path
sys.path.append(LIB_PATH)
# from geo_tools import rd_helper

pyglet.options['shadow_window'] = False


# colors
cs = {
    'blue_bright': [0.0, 0.3, 1.0],
    'orange': [1.0, 0.3, 0.0],
    'red': [1.0, 0.0, 0.0],
    'blue': [0.0, 0.0, 1.0],
    'purple': [0.6, 0.0, 0.8]
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
    y_rot = [
        [c,  0,  s, 0],
        [0,  1,  0, 0],
        [-s,  0,  c, 0],
        [0,  0,  0, 1]
    ]
    self.apply_transform(y_rot)
    return self


trimesh.Trimesh.rotate_in_yaxis = rotate_in_yaxis


def load_bunny(rotation_degrees='000'):
    file_name = "bun{}_v2.ply".format(rotation_degrees)
    mesh_file_path = os.path.join(RES_PATH, file_name)
    assert os.path.exists(mesh_file_path), 'cannot find:' + file_name
    tmesh = trimesh.load(mesh_file_path)
    return tmesh


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


def run_gui_pyrmesh(pyr_meshes):
    scene = generateSceneWithPyrenderMeshes(pyr_meshes)
    v = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=10)
    del v


def get_offline_render(tmeshes):
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    camera_pose = np.array([
        [1,  0,   0,    0],
        [0,  1,   0,  0.1],
        [0,  0,   1,  0.3],
        [0,  0,   0,    1]
    ])

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
    plt.axis('off')
    plt.show()


def color_overlaps(tm1, tm2, color_1, color_2, color_overlap, metric='median'):

    def choose_color(dist, threshold, col_close, col_far):

        if metric == 'median':
            isclose = dist < threshold
        elif metric == 'abs':
            isclose = dist < 0.005
        else:
            isclose = dist < 0.005

        if isclose:
            color = col_close
        else:
            color = col_far
        return color

    def color_overlaps_in_one_mesh(
            to_color_tmesh, reference_tmesh,
            color_overlap, color_non_overlap):

        closest_points_to_find = 5

        P = np.array(reference_tmesh.vertices)
        Q = np.array(to_color_tmesh.vertices)

        tree_P = KDTree(P)

        PQ_dists, _ = tree_P.query(Q, k=closest_points_to_find)
        PQ_dists = np.mean(PQ_dists, 1)

        median_PQ = np.median(PQ_dists)
        to_color_tmesh.visual.vertex_colors = [
            choose_color(dist, median_PQ * 3, color_overlap, color_non_overlap)
            for dist
            in PQ_dists
        ]

    color_overlaps_in_one_mesh(tm1, tm2, color_overlap, color_1)
    color_overlaps_in_one_mesh(tm2, tm1, color_overlap, color_2)


"""
The function assumes the meshes overlap largely
"""


def visualise_overlaps(tm1, tm2, metric='median'):

    tm1_to_color = tm1.copy()
    tm2_to_color = tm2.copy()

    color_overlaps(tm1_to_color, tm2_to_color,
                   cs['red'], cs['blue'], cs['blue_bright'], metric=metric)

    tm1_org = tm1.copy().color_verts(
        cs['red']).apply_translation([0.1, 0.12, -0.15])
    tm2_org = tm2.copy().color_verts(
        cs['blue']).apply_translation([0.1, -0.12, -0.15])

    run_gui_tmesh([tm1_to_color, tm2_to_color, tm1_org, tm2_org])

# ICP


def subsamplePointsRandomly(points, subset_size=1):
    number_of_points = points.shape[0]
    sampled_indexes = np.random.choice(
        number_of_points,
        int(number_of_points * subset_size),
        replace=False)
    points_sampled = points[sampled_indexes]
    return points_sampled, sampled_indexes


"""
    tm1 is fixed
    tm2 is moving

    while not convergred:
        Q = np.array(tm2.vertices)

        1. subsample from Q
        2. for each point in Q, find closest point in P
            query the tree for closest point 
        3. scrap outliers
        4. find the transofrm 
            p_dash = np.mean(P)
            q_dash = np.mean(Q)
            t = p_dash - R * q_dash

            calculate R from the SVD
        5. check convergance

    @params
    verbose = {0,1,2}
"""


def icp(tm1,
        tm2,
        KDTree_mesh1,
        rejection=True,
        early_stopping=False,
        verbose=1,
        verbose_pic=0,
        verbose_pic_final=0,
        random_subsampling=0.3):

    def print_info(i, error, delta):
        print("Iteration: {0} \t error: {1:.4f} \t delta: {2:.4f}".format(
            i+1, error, delta))

    if verbose_pic:
        run_offline_tmesh([tm1, tm2])

    if verbose > 0:
        print("Starting ICP point-to-point")

    errors = []
    deltas = []

    P = np.array(tm1.vertices)

    max_iterations = 50
    for i in range(max_iterations):

        Q = np.array(tm2.vertices)

        # subsample from Q
        Q_sam, _ = subsamplePointsRandomly(Q, random_subsampling)

        # get corresondings points
        PQ_dists, P_corresponding_inds = KDTree_mesh1.query(Q_sam)

        # remove outliers
        # criterium: remove points further than median
        if rejection:
            median = np.median(PQ_dists)
            PQ_pairs_inds = np.array([
                [P_ind, Q_ind]
                for Q_ind, (dist, P_ind)
                in enumerate(zip(PQ_dists[:, 0], P_corresponding_inds[:, 0]))
                if dist <= median * 3
            ])

            P_filtered = P[PQ_pairs_inds[:, 0]]
            Q_filtered = Q_sam[PQ_pairs_inds[:, 1]]
        else:
            P_filtered = P[P_corresponding_inds[:, 0]]
            Q_filtered = Q_sam

        # calculate R, t
        p_fil_mean = np.mean(P_filtered, 0)
        q_fil_mean = np.mean(Q_filtered, 0)

        P_hat = P_filtered - p_fil_mean
        Q_hat = Q_filtered - q_fil_mean

        A = np.sum([
            np.matmul(q_hat_i.reshape(3, 1), p_hat_i.reshape(1, 3))
            for (p_hat_i, q_hat_i)
            in zip(P_hat, Q_hat)
        ], 0)

        u, s, vh = np.linalg.svd(A)
        utv = np.matmul(vh.T, u.T)
#         print(np.linalg.det(utv))

        R = utv
        t = p_fil_mean.reshape(3, 1) - np.matmul(R, q_fil_mean.reshape(3, 1))

        transformation = np.eye(4, 4)
        transformation[:3, :3] = R
        transformation[:3, 3] = t[:, 0]

        tm2.apply_transform(transformation)

        error = np.sum([
            np.linalg.norm(p.reshape(3, 1) - np.matmul(R,
                                                       q.reshape(3, 1)) - t.reshape(3, 1), 2) ** 2
            for (p, q)
            in zip(P_filtered, Q_filtered)
        ])

        errors.append(error)

        delta = 0
        if i > 0:
            delta = errors[-1] - errors[-2]
            deltas.append(delta)

        if verbose == 2:
            print_info(i, error, delta)

        if verbose == 1:
            if (i+1) % 10 == 0:
                print_info(i, error, delta)
                if verbose_pic:
                    run_offline_tmesh([tm1, tm2])

#         if early_stopping and i > 0 and err[-1] / err[-2] > 1.1:
        if early_stopping and delta > 0.05:
            print("Early stopping after {}/{} iterations".format(i+1, max_iterations))
            break

    return errors, deltas


def plot_log(errors, deltas, comment=""):

    ax = plt.subplot(111, label="1")
    ax.set_title("ICP results\n{}".format(comment))
    ax.set_xlabel("Iteration")
    ax.plot(
        np.arange(1, len(errors)+1, 1),
        [np.math.log(x) for x in errors],
        label='log(error)')
    ax.plot(
        np.arange(2, len(deltas)+2, 1),
        [np.math.log(abs(x)) for x in deltas],
        label='log(abs(error delta))')
    ax.legend()
    plt.show()


def perform_icp_for_rotated_mesh(tmesh_fixed, angle_yaxis, tree_fixed):
    tmesh_rotated = tmesh_fixed \
        .copy() \
        .rotate_in_yaxis(angle_yaxis) \
        .color_verts(cs['blue_bright'])

    errors, deltas = icp(tmesh_fixed,
                         tmesh_rotated,
                         KDTree_mesh1=tree_fixed,
                         verbose=0,
                         verbose_pic=0,
                         random_subsampling=0.003)

    return tmesh_rotated, errors, deltas


def get_noisy_mesh(tmesh, constant):
    tm_noisy = tmesh.copy()

    tm_bounds_mean_range = np.mean(
        abs(tm_noisy.bounds[0, :] - tm_noisy.bounds[1, :]))
    noise_multiplier = constant * tm_bounds_mean_range

    tm_noisy.vertices += np.random.randn(*
                                         tm_noisy.vertices.shape) * noise_multiplier

    return tm_noisy


def color_with_normal_shading(tmesh):
    tmesh.visual.vertex_colors = (tmesh.vertex_normals + 1) / 2
    return tmesh


"""
    tm1 is fixed
    tm2 is moving

    while not convergred:
        Q = np.array(tm2.vertices)

        1. subsample from Q
        2. for each point in Q, find closest point in P
            query the tree for closest point 
        3. scrap outliers
        4. find the transofrm 
            p_dash = np.mean(P)
            q_dash = np.mean(Q)
            t = p_dash - R * q_dash

            calculate R from the SVD
        5. check convergance

    @params
    verbose = {0,1,2}
"""


def icp_point_to_plane(tm1,
                       tm2,
                       KDTree_mesh1,
                       rejection=True,
                       early_stopping=False,
                       verbose=1,
                       verbose_pic=0,
                       verbose_pic_final=0,
                       random_subsampling=0.2):

    def print_info(i, error, delta):
        print("Iteration: {0} \t error: {1:.4f} \t delta: {2:.4f}".format(
            i+1, error, delta))

    if verbose_pic:
        run_offline_tmesh([tm1, tm2])

    errors = []
    deltas = []

    P = np.array(tm1.vertices)

    print("Starting ICP point-to-plane")

    max_iterations = 50
    for i in range(max_iterations):

        Q = np.array(tm2.vertices)

        # subsample from Q
        Q_sam, Q_sam_ind = subsamplePointsRandomly(Q, random_subsampling)

        # get corresondings points
        PQ_dists, P_corresponding_inds = KDTree_mesh1.query(Q_sam)

        # remove outliers
        # criterium: remove points further than median
        if rejection:
            median = np.median(PQ_dists)
            PQ_pairs_inds = np.array([
                [P_ind, Q_ind]
                for Q_ind, (dist, P_ind)
                in enumerate(zip(PQ_dists[:, 0], P_corresponding_inds[:, 0]))
                if dist <= median * 3
            ])

            P_filtered = P[PQ_pairs_inds[:, 0]]
            Q_filtered = Q_sam[PQ_pairs_inds[:, 1]]

        else:
            P_filtered = P[P_corresponding_inds[:, 0]]
            Q_filtered = Q_sam

        # calculate r, t
        P_filtered_normals = tm1.vertex_normals[PQ_pairs_inds[:, 0]]

        number_of_points = P_filtered.shape[0]
        A = np.zeros((number_of_points, 6))
        A[:, :3] = np.cross(Q_filtered, P_filtered_normals)
        A[:, 3:] = P_filtered_normals

        b = np.array([
            - np.dot((q-p), n)
            for (q, p, n)
            in zip(Q_filtered, P_filtered, P_filtered_normals)
        ])

        u, s_flat, vt = np.linalg.svd(A)
        s_flat_ps_inv = [1/x for x in s_flat]
        temp = np.zeros_like(A.T)
        temp[:len(s_flat), :len(s_flat)] = np.diag(s_flat_ps_inv)
        s_ps_inv = temp
        A_ps_inv = vt.T @ s_ps_inv @ u.T

        solution = A_ps_inv @ b

        ra, rb, rg, tx, ty, tz = solution

        transformation = np.eye(4)
        R = Rotation.from_rotvec([ra, rb, rg]).as_dcm()
        transformation[:3, :3] = R

        t = np.array([tx, ty, tz])
        transformation[:3, 3] = t

        tm2.apply_transform(transformation)

#         error = np.sum([
#             np.dot((R@q + t - p), n) ** 2
#             for (q,p,n)
#             in zip(Q_filtered, P_filtered, P_filtered_normals)
#         ])

        # for inspection and comparison, use metric from point-to-point
        error = np.sum([
            np.linalg.norm(p.reshape(3, 1) - np.matmul(R,
                                                       q.reshape(3, 1)) - t.reshape(3, 1), 2) ** 2
            for (p, q)
            in zip(P_filtered, Q_filtered)
        ])

        errors.append(error)

        delta = 0
        if i > 0:
            delta = errors[-1] - errors[-2]
            deltas.append(delta)

        if verbose == 2:
            print_info(i, error, delta)

        if verbose == 1:
            if (i+1) % 10 == 0:
                print_info(i, error, delta)
                if verbose_pic:
                    run_offline_tmesh([tm1, tm2])

#         if early_stopping and i > 0 and err[-1] / err[-2] > 1.1:
        if early_stopping and i > 0 and abs(delta) < 0.0001:
            print("Early stopping after {}/{} iterations".format(i+1, max_iterations))
            break

    return errors, deltas


def icp_many(meshes,
             rejection=True,
             early_stopping=False,
             verbose=1,
             verbose_pic=0,
             verbose_pic_final=0,
             random_subsampling=0.01):

    def print_info(i, error, delta):
        print("Iteration: {0} \t error: {1:.4f} \t delta: {2:.4f}".format(
            i+1, error, delta))

    if verbose > 0:
        print("Starting ICP point-to-point many meshes")

    errors = []
    deltas = []

    for k in range(150):

        for i, mesh_parent in enumerate(meshes):

            P = np.array(mesh_parent.vertices)
            KDTree_mesh1 = KDTree(P)

            for j, mesh_child in enumerate(meshes):

                if (j == i):
                    continue

                tm2 = mesh_child

                max_iterations = 5
                for i in range(max_iterations):

                    Q = np.array(tm2.vertices)

                    # subsample from Q
                    Q_sam, _ = subsamplePointsRandomly(Q, random_subsampling)

                    # get corresondings points
                    PQ_dists, P_corresponding_inds = KDTree_mesh1.query(Q_sam)

                    # remove outliers
                    # criterium: remove points further than median
                    if rejection:
                        median = np.median(PQ_dists)
                        PQ_pairs_inds = np.array([
                            [P_ind, Q_ind]
                            for Q_ind, (dist, P_ind)
                            in enumerate(zip(PQ_dists[:, 0], P_corresponding_inds[:, 0]))
                            if dist <= median * 3
                        ])

                        P_filtered = P[PQ_pairs_inds[:, 0]]
                        Q_filtered = Q_sam[PQ_pairs_inds[:, 1]]
                    else:
                        P_filtered = P[P_corresponding_inds[:, 0]]
                        Q_filtered = Q_sam

                    # calculate R, t
                    p_fil_mean = np.mean(P_filtered, 0)
                    q_fil_mean = np.mean(Q_filtered, 0)

                    P_hat = P_filtered - p_fil_mean
                    Q_hat = Q_filtered - q_fil_mean

                    A = np.sum([
                        np.matmul(q_hat_i.reshape(3, 1), p_hat_i.reshape(1, 3))
                        for (p_hat_i, q_hat_i)
                        in zip(P_hat, Q_hat)
                    ], 0)

                    u, s, vh = np.linalg.svd(A)
                    utv = np.matmul(vh.T, u.T)
            #         print(np.linalg.det(utv))

                    R = utv
                    t = p_fil_mean.reshape(
                        3, 1) - np.matmul(R, q_fil_mean.reshape(3, 1))

                    transformation = np.eye(4, 4)
                    transformation[:3, :3] = R
                    transformation[:3, 3] = t[:, 0]

                    tm2.apply_transform(transformation)

    run_offline_tmesh(meshes)
    #
