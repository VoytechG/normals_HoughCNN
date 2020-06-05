import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import pyrender
import matplotlib.pyplot as plt

from mesh_tools import run_gui_pyrmesh


class HoughEstimator:
    def __init__(self, point_cloud_path, K_multipliers, ground_truth_normals=None):

        # Distance from origin of points for which the accumulator is calculated
        self.max_dist_to_origin_sq = 0.02

        # Accumulator (33x33) side size
        self.A = 33

        # size of neighbourhood of the point being estimated
        self.K = 100
        self.Ks = np.array(self.K * np.sort(K_multipliers), dtype="uint")

        # Number of hypthesis drawn for each sampled point
        self.T = 1000

        # nth furthest point to take into account in density variation trick
        self.aniso_th = 5

        self.point_cloud = None
        self.load_point_cloud(point_cloud_path)

        self.ground_truth_normals = ground_truth_normals

        self.VISUALISE_HYPOTHESIS = False

    def K_len(self):
        return len(self.Ks)

    def load_point_cloud(self, file_path):
        lines = open(file_path).read().split("\n")[:-1]
        self.point_cloud = np.array(
            [[float(x) for x in line.split(" ")[:3]] for line in lines]
        )
        return self

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
        if normal[2] < 0:
            # reorient normal
            normal = normal * -1
        return normal

    def create_accumulator_with_normals(self, normals):

        accumulator = np.zeros((self.A, self.A))

        for normal in normals:
            x = int(np.round((normal[0] + 1) / 2 * self.A)) - 1
            y = int(np.round((normal[1] + 1) / 2 * self.A)) - 1
            accumulator[x, y] += 1

        return accumulator

    def visualise_hypothesis_estimation(
        self,
        point_cloud,
        neighbourhood_cloud_indices,
        hypothesised_point_index,
        sampled_three_points_indices,
        hypothesised_normal,
    ):
        color_values = {
            "point_cloud": [255, 255, 255],
            "neighbourhood_cloud": [235, 189, 52],
            "hypothesised_point": [182, 0, 214],
            "sampled_three_points": [232, 0, 0],
            "estimated_normal": [86, 0, 214],
        }

        hypothesised_point = point_cloud[hypothesised_point_index]

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
                                hypothesised_point + hypothesised_normal * 0.12,
                            ],
                            mode=3,
                            color_0=normal_colors,
                        ),
                        pyrender.Primitive([a, b], mode=3, color_0=triangle_colors),
                        pyrender.Primitive([b, c], mode=3, color_0=triangle_colors),
                        pyrender.Primitive([c, a], mode=3, color_0=triangle_colors),
                    ]
                ),
            ],
            point_size=10,
        )

    def visualise_accumulator(self, accumulator_no_pcas, accumulator_after_3d_pca):
        # plt.figure()

        # subplot(r,c) provide the no. of rows and columns
        # _, subplots = plt.subplots(1, 2)

        # # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        # subplots[0].imshow(accumulator_no_pcas)
        # subplots[1].imshow(accumulator_after_3d_pca)
        plt.subplot(1, 2, 1)
        plt.imshow(accumulator_no_pcas)
        plt.title("No PCA applied")

        plt.subplot(1, 2, 2)
        plt.imshow(accumulator_after_3d_pca)
        plt.title("3D PCA applied")

        plt.show()

    def generate_accums_for_point_cloud(self, number_of_points):

        accumulators = np.zeros((number_of_points, len(self.Ks), self.A, self.A))

        if self.ground_truth_normals is None:
            target_normals = None
        else:
            target_normals = np.zeros((number_of_points, 3))

        point_cloud = self.point_cloud
        points_close_to_origin_indices = self.get_points_close_to_origin()

        # visualise_valid_points(point_cloud, points_close_to_origin_indices)

        kd_tree = KDTree(point_cloud)
        K_max = self.Ks[-1]

        # +1 to compensate for the fact that the first result is the point itself
        distances, indices = kd_tree.query(
            point_cloud, k=max(self.aniso_th + 1, K_max + 1)
        )

        # Robust prob. distribution (no 5 bins "discretised" trick)
        # Paper fig. 8c
        aniso_probabilities = np.array(distances[:, self.aniso_th])

        if len(points_close_to_origin_indices) > number_of_points:
            points_close_to_origin_indices = points_close_to_origin_indices[
                :number_of_points
            ]

        for i, point_index in enumerate(points_close_to_origin_indices):

            max_neighbourhod_point_cloud = point_cloud[indices[point_index]]

            R_pca_3d = self.get_pca_axis_3d(max_neighbourhod_point_cloud)
            # Aligns z-axis on the smallest Principal Component
            # point_cloud_rotated = self.rotate_point_cloud(point_cloud, pca_rot_3d)

            accumulators_of_sampled_point = np.zeros((len(self.Ks), self.A, self.A))

            for k_index, k in enumerate(self.Ks):
                accum_normals = np.zeros((k, 3))

                local_aniso_prob = aniso_probabilities[indices[point_index, : k + 1]]
                # 0 prob of choosing the hypothesised point
                local_aniso_prob[0] = 0
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

                accumulator_no_pcas = self.create_accumulator_with_normals(normals)

                normals_after_PCA_3D = np.array(
                    [R_pca_3d @ normal for normal in normals]
                )
                accumulator_after_3d_pca = self.create_accumulator_with_normals(
                    normals_after_PCA_3D
                )

                self.visualise_accumulator(
                    accumulator_no_pcas, accumulator_after_3d_pca
                )

                # accumulators_k = self.create_accumulator_with_normals(normals)

                # # TODO add 2nd PCA

                # if self.VISUALISE_HYPOTHESIS and hypothesis_index < 1:
                #     print(f"Index of hypo point {point_index}")
                #     self.visualise_hypothesis_estimation(
                #         point_cloud=point_cloud_rotated[
                #             indices[point_index, 0 : K_max + 1]
                #         ],
                #         neighbourhood_cloud_indices=np.arange(1, k + 1),
                #         hypothesised_point_index=0,
                #         sampled_three_points_indices=[ia, ib, ic],
                #         hypothesised_normal=normal,
                #     )

            # accumulators[i] = accumulators_of_sampled_point

            if target_normals is not None:
                target_normals[i] = self.ground_truth_normals[point_index]

        return accumulators, target_normals
