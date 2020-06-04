import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import pyrender

from implementations import run_gui_pyrmesh


class HoughEstimator:
    def __init__(self, point_cloud_path, K_multipliers, ground_truth_normals=None):

        # Distance from origin of points for which the accumulator is calculated
        self.MAX_DIST_TO_ORIGIN_SQ = 0.02

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
        max_dst = self.MAX_DIST_TO_ORIGIN_SQ
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
            "estimated_normal": [86, 0, 214, 255],
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

        normal_colors = np.ones((2, 4)) * color_values["estimated_normal"] / 255

        run_gui_pyrmesh(
            [
                pyrender.Mesh.from_points(point_cloud, colors=colors),
                # pyrender.Primitive(
                #     [hypothesised_point, hypothesised_point + hypothesised_normal],
                #     mode=3,
                #     color_0=normal_colors,
                # ),
            ],
            point_size=5,
        )

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
        # Paper fig. 8c)
        aniso_probabilities = np.array(distances[:, self.aniso_th])

        if len(points_close_to_origin_indices) > number_of_points:
            points_close_to_origin_indices = points_close_to_origin_indices[
                :number_of_points
            ]

        for i, point_index in enumerate(points_close_to_origin_indices):

            max_neighbourhod_point_cloud = point_cloud[indices[point_index]]

            pca_rot_3d = self.get_pca_axis_3d(max_neighbourhod_point_cloud)
            # Aligns z-axis on the smallest Principal Component
            point_cloud_rotated = self.rotate_point_cloud(point_cloud, pca_rot_3d)

            # shape (len(self.Ks), 33, 33)
            accumulators_of_sampled_point = np.zeros(accumulators.shape[1:])

            for k_index, k in enumerate(self.Ks):
                accum_normals = np.zeros((k, 3))

                local_aniso_prob = aniso_probabilities[indices[point_index, : k + 1]]
                # 0 prob of choosing the hypothesised point
                local_aniso_prob[0] = 0
                local_aniso_prob /= np.sum(local_aniso_prob)

                random_choices = np.random.choice(
                    k + 1, (self.T, 3), replace=True, p=local_aniso_prob
                )

                for hypothesis_index, [ia, ib, ic] in enumerate(random_choices):

                    if ia == ib or ib == ic or ia == ic:
                        [ia, ib, ic] = np.random.choice(
                            k + 1, (3), replace=False, p=local_aniso_prob
                        )

                    [a, b, c] = point_cloud_rotated[indices[point_index, [ia, ib, ic]]]

                    normal = np.cross(b - a, c - a)
                    normal = normal / np.linalg.norm(normal)
                    if normal[2] < 0:
                        # reorient normal
                        normal = normal * -1

                    x = int(np.round((normal[0] + 1) / 2 * self.A)) - 1
                    y = int(np.round((normal[1] + 1) / 2 * self.A)) - 1

                    accumulators_of_sampled_point[k_index, x, y] += 1

                    # TODO add 2nd PCA

                    if self.VISUALISE_HYPOTHESIS and hypothesis_index < 1:
                        print(f"Index of hypo point {point_index}")
                        self.visualise_hypothesis_estimation(
                            point_cloud=point_cloud_rotated,
                            neighbourhood_cloud_indices=indices[point_index, 1 : k + 1],
                            hypothesised_point_index=point_index,
                            sampled_three_points_indices=indices[
                                point_index, [ia, ib, ic]
                            ],
                            hypothesised_normal=normal,
                        )

            accumulators[i] = accumulators_of_sampled_point

            if target_normals is not None:
                target_normals[i] = self.ground_truth_normals[point_index]

        return accumulators, target_normals
