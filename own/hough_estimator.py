import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import pyrender
import matplotlib.pyplot as plt
from enum import Enum, auto

from mesh_tools import run_gui_pyrmesh, get_axis_trimesh


class HoughEstimator:
    def __init__(self, number_of_channels, point_cloud=None, ground_truth_normals=None):

        self.ground_truth_normals = ground_truth_normals
        self.VISUALISE_HYPOTHESIS = False
        self.VISUALISE_ACCUMULATOR = False
        self.VISUALISE_VALID_POINTS = False

        # Distance from origin of points for which the accumulator is calculated
        # self.max_dist_to_origin_sq = 0.2
        self.points_to_subsample_multiplier = 3

        # Accumulator (33x33) side size
        self.A = 33

        # size of neighbourhood of the point being estimated
        self.K = 100
        if number_of_channels == 3:
            self.Ks = np.array([self.K, self.K // 2, self.K * 2])
        else:
            self.Ks = np.array([self.K])

        # Number of hypthesis drawn for each sampled point
        self.T = 1000

        # nth furthest point to take into account in density variation trick
        self.aniso_th = 5

        self.point_cloud = point_cloud

    @classmethod
    def load_point_cloud(cls, file_path):
        lines = open(file_path).read().split("\n")[:-1]
        point_cloud = np.array(
            [[float(x) for x in line.split(" ")[:3]] for line in lines]
        )
        return point_cloud

    def K_len(self):
        return len(self.Ks)

    @DeprecationWarning
    def get_points_close_to_origin(self):
        point_cloud = self.point_cloud
        max_dst = self.max_dist_to_origin_sq
        return np.array(
            [
                i
                for i, point in enumerate(point_cloud)
                if np.linalg.norm(point) < max_dst ** 0.5
            ]
        )

    def get_pca_axis_3d(self, point_cloud_subset):
        pca = PCA(n_components=3)
        pca.fit(point_cloud_subset)
        return pca.components_

    def rotate_point_cloud(self, point_cloud_subset, rot_matrix):
        return np.array((rot_matrix @ point_cloud_subset.T).T)

    def generate_random_triplet_choices(self, max_point_index, probabilities):

        random_choices = np.random.choice(
            max_point_index, (self.T, 3), replace=True, p=probabilities
        )

        random_choices = [
            np.random.choice(max_point_index, (3), replace=False, p=probabilities)
            if ia == ib or ib == ic or ia == ic
            else [ia, ib, ic]
            for [ia, ib, ic] in random_choices
        ]

        return random_choices

    def get_normal_from_three_points(self, a, b, c):
        normal = np.cross(b - a, c - a)
        normal = normal / np.linalg.norm(normal)
        return normal

    def get_accumulator_coordaintes(self, normal):
        x = int(np.round((normal[0] + 1) / 2 * self.A)) - 1
        y = int(np.round((normal[1] + 1) / 2 * self.A)) - 1
        return x, y

    def create_accumulator_with_normals(self, normals):

        accumulator = np.zeros((self.A, self.A))

        for normal in normals:

            x, y = self.get_accumulator_coordaintes(normal)
            accumulator[y, x] += 1

        return accumulator

    def apply_full_normalization(self, normal, R_pca_3d, R_pca_2d, return_steps=False):
        normal_init = normal
        normal_after_3d_r_pca = R_pca_3d @ normal_init
        if normal_after_3d_r_pca[2] < 0:
            normal_after_3d_r_pca *= -1
        normal_after_2d_r_pca = R_pca_2d @ normal_after_3d_r_pca

        if return_steps:
            return normal_init, normal_after_3d_r_pca, normal_after_2d_r_pca

        return normal_after_2d_r_pca

    def visualise_valid_points(
        self, points, sampling_set_indices, sampled_subset_indices
    ):
        colors = np.ones_like(points)
        colors[sampling_set_indices] = [1, 0.3, 0.1]
        colors[sampled_subset_indices] = [0.1, 1, 0.4]

        pymesh = pyrender.Mesh.from_points(points, colors=colors)
        run_gui_pyrmesh([pymesh], point_size=8)

    def visualise_hypothesis_estimation(
        self,
        point_cloud,
        neighbourhood_cloud_indices,
        hypothesised_point_index,
        sampled_three_points_indices,
        ground_truth_normal_index,
    ):
        color_values = {
            "point_cloud": [255, 255, 255],
            "neighbourhood_cloud": [235, 189, 52],
            "hypothesised_point": [182, 0, 214],
            "sampled_three_points": [232, 0, 0],
            "estimated_normal": [86, 0, 214],
        }

        hypothesised_point = point_cloud[hypothesised_point_index]
        hypothesised_normal = self.get_normal_from_three_points(
            *point_cloud[sampled_three_points_indices]
        )
        true_normal = self.ground_truth_normals[ground_truth_normal_index]

        colors = (
            np.ones((len(point_cloud), 3)) * np.array(color_values["point_cloud"]) / 255
        )
        colors[neighbourhood_cloud_indices] = (
            np.array(color_values["neighbourhood_cloud"]) / 255
        )
        colors[hypothesised_point_index] = (
            np.array(color_values["hypothesised_point"]) / 255
        )
        colors[sampled_three_points_indices] = (
            np.array(color_values["sampled_three_points"]) / 255
        )

        normal_colors = (
            np.ones((2, 4)) * (color_values["estimated_normal"] + [255]) / 255
        )
        triangle_colors = (
            np.ones((2, 4)) * (color_values["sampled_three_points"] + [255]) / 255
        )
        [a, b, c] = point_cloud[sampled_three_points_indices]

        run_gui_pyrmesh(
            [
                pyrender.Mesh.from_points(point_cloud, colors=colors),
                pyrender.Mesh(
                    [
                        pyrender.Primitive(
                            [
                                hypothesised_point,
                                hypothesised_point + true_normal * 0.12,
                            ],
                            mode=3,
                            color_0=normal_colors,
                        ),
                        pyrender.Primitive(
                            [
                                hypothesised_point,
                                hypothesised_point + hypothesised_normal * 0.12,
                            ],
                            mode=3,
                            color_0=triangle_colors,
                        ),
                        pyrender.Primitive([a, b], mode=3, color_0=triangle_colors),
                        pyrender.Primitive([b, c], mode=3, color_0=triangle_colors),
                        pyrender.Primitive([c, a], mode=3, color_0=triangle_colors),
                    ]
                ),
            ],
            point_size=10,
        )

    def visualise_accumulator(
        self,
        normals,
        normals_after_PCA_3D,
        normals_after_PCA_2D,
        ground_truth_normal=None,
        R_pca_3d=None,
        R_pca_2d=None,
        block=True,
    ):
        accumulator_no_pcas = self.create_accumulator_with_normals(normals)
        accumulator_after_3d_pca = self.create_accumulator_with_normals(
            normals_after_PCA_3D
        )
        accumulator_after_2d_pca = self.create_accumulator_with_normals(
            normals_after_PCA_2D
        )

        if ground_truth_normal is not None:
            n, n_3dr, n_2dr = self.apply_full_normalization(
                ground_truth_normal, R_pca_3d, R_pca_2d, return_steps=True
            )

        n_inv = R_pca_3d.T @ R_pca_2d.T @ n_2dr
        [n, n_3dr, n_2dr, n_inv] = [
            self.get_accumulator_coordaintes(normal)
            for normal in [n, n_3dr, n_2dr, n_inv]
        ]

        plt.clf()

        plt.subplot(1, 3, 1)
        plt.imshow(accumulator_no_pcas, cmap="gray")
        plt.title("No PCA")
        if ground_truth_normal is not None:
            plt.scatter(x=n[0], y=n[1], c="r", s=5)

        plt.subplot(1, 3, 2)
        plt.imshow(accumulator_after_3d_pca, cmap="gray")
        plt.title("3D PCA & flipping")
        if ground_truth_normal is not None:
            plt.scatter(x=n_3dr[0], y=n_3dr[1], c="r", s=5)

        plt.subplot(1, 3, 3)
        plt.imshow(accumulator_after_2d_pca, cmap="gray")
        plt.title("3D PCA, flipping, 2D PCA")
        if ground_truth_normal is not None:
            plt.scatter(x=n_2dr[0], y=n_2dr[1], c="r", s=5)
            # plt.scatter(x=n_inv[0], y=n_inv[1], c="r", s=5)

        if block:
            plt.show()
        else:
            plt.ion()
            plt.show()

    def generate_accums_for_point_cloud(self, batch_size):

        point_cloud = self.point_cloud
        kd_tree = KDTree(point_cloud)

        number_of_points_to_sample = min(
            batch_size * self.points_to_subsample_multiplier, len(point_cloud)
        )

        sampling_set_indices = kd_tree.query(
            [[0, 0, 0]], number_of_points_to_sample, return_distance=False
        )[0]

        sampled_points_indices = sampling_set_indices[
            np.random.choice(number_of_points_to_sample, batch_size, replace=False)
        ]

        if self.VISUALISE_VALID_POINTS:
            self.visualise_valid_points(
                point_cloud,
                sampling_set_indices=sampling_set_indices,
                sampled_subset_indices=sampled_points_indices,
            )

        accumulators = np.zeros((batch_size, len(self.Ks), self.A, self.A))
        inverse_rotation_matrices = np.zeros((batch_size, len(self.Ks), 3, 3))

        if self.ground_truth_normals is not None:
            target_normals = np.zeros((batch_size, 2))
        else:
            target_normals = None

        K_max = np.max(self.Ks)

        # +1 to compensate for the fact that the first result is the point itself
        distances, indices = kd_tree.query(
            point_cloud, k=max(self.aniso_th + 1, K_max + 1)
        )

        # Robust prob. distribution (no 5 bins "discretised" trick), Paper fig. 8c
        aniso_probabilities = np.array(distances[:, self.aniso_th])

        for i, point_index in enumerate(sampled_points_indices):

            K_neighbourhood_indices = indices[point_index, : self.K + 1]

            R_pca_3d = self.get_pca_axis_3d(
                np.array(
                    [
                        list(point * prob)
                        for point, prob in zip(
                            point_cloud[K_neighbourhood_indices],
                            aniso_probabilities[K_neighbourhood_indices],
                        )
                    ]
                )
            )
            R_pca_2d = None

            # point_cloud_rotated = self.rotate_point_cloud(point_cloud, pca_rot_3d)

            accumulators_of_sampled_point = np.zeros((len(self.Ks), self.A, self.A))

            for k_index, k in enumerate(self.Ks):
                accum_normals = np.zeros((k, 3))

                neighbourhood_indices = indices[point_index, : k + 1]

                local_aniso_prob = aniso_probabilities[neighbourhood_indices]
                local_aniso_prob[0] = 0  # 0 prob of choosing the hypothesised point
                local_aniso_prob /= np.sum(local_aniso_prob)

                random_choices = self.generate_random_triplet_choices(
                    k + 1, local_aniso_prob
                )

                normals = np.zeros((len(random_choices), 3))

                for hypothesis_index, [ia, ib, ic] in enumerate(random_choices):
                    [a, b, c] = point_cloud[indices[point_index, [ia, ib, ic]]]

                    normals[hypothesis_index] = self.get_normal_from_three_points(
                        a, b, c
                    )

                normals_after_PCA_3D = (R_pca_3d @ normals.T).T
                normals_after_PCA_3D = np.array(
                    [
                        normal if normal[2] >= 0 else normal * -1
                        for normal in normals_after_PCA_3D
                    ]
                )

                if R_pca_2d is None:
                    R_pca_2d = self.get_pca_axis_3d(
                        np.array(
                            [
                                [normal[0], normal[1], 0]
                                for normal in normals_after_PCA_3D
                            ]
                        )
                    )

                normals_after_PCA_2D = (R_pca_2d @ normals_after_PCA_3D.T).T

                if self.VISUALISE_ACCUMULATOR:

                    if self.ground_truth_normals is not None:
                        ground_truth_normal = self.ground_truth_normals[point_index]
                    else:
                        ground_truth_normal = None

                    self.visualise_accumulator(
                        normals,
                        normals_after_PCA_3D,
                        normals_after_PCA_2D,
                        ground_truth_normal,
                        R_pca_3d,
                        R_pca_2d,
                        block=False,
                    )

                    self.visualise_valid_points(
                        point_cloud[indices[point_index]], np.arange(0, k + 1), [0],
                    )

                    if self.VISUALISE_HYPOTHESIS:
                        print(f"Index of hypo point {point_index}")
                        self.visualise_hypothesis_estimation(
                            point_cloud[indices[point_index]],
                            neighbourhood_cloud_indices=np.arange(1, k + 1),
                            hypothesised_point_index=0,
                            sampled_three_points_indices=[ia, ib, ic],
                            ground_truth_normal_index=point_index,
                        )

                accumulators_of_sampled_point[
                    k_index
                ] = self.create_accumulator_with_normals(normals_after_PCA_2D)

            accumulators[i] = accumulators_of_sampled_point
            inverse_rotation_matrices[i] = R_pca_3d.T @ R_pca_2d.T

            if target_normals is not None:
                target_normals[i] = self.apply_full_normalization(
                    self.ground_truth_normals[point_index], R_pca_3d, R_pca_2d
                )[:2]

        if target_normals is not None:
            return accumulators, inverse_rotation_matrices, target_normals

        return accumulators, inverse_rotation_matrices
