import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree

from implementations import run_gui_pyrmesh


class HoughEstimator:
    def __init__(self, point_cloud_path, K_multipliers):

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

    def generate_accums_for_point_cloud(self, number_of_points):

        point_cloud = self.point_cloud

        points_close_to_origin_indices = self.get_points_close_to_origin()

        # visualise_valid_points(point_cloud, points_close_to_origin_indices)

        kd_tree = KDTree(point_cloud)
        K_max = self.Ks[-1]

        # +1 to compensate for the fact that the first result is the point itself
        distances, indices = kd_tree.query(
            point_cloud, k=max(self.aniso_th + 1, K_max + 1)
        )
        aniso_probabilities = np.array(distances[:, 5])

        if len(points_close_to_origin_indices) > number_of_points:
            points_close_to_origin_indices = points_close_to_origin_indices[
                :number_of_points
            ]

        for point_index in points_close_to_origin_indices:
            point = point_cloud[point_index]

            max_neighbourhod_indices = indices[point_index, 1 : K_max + 1]
            max_neighbourhod = point_cloud[max_neighbourhod_indices]

            pca_rot_3d = self.get_pca_axis_3d(max_neighbourhod)
            # Aligns z-axis on the smallest Principal Component
            max_neighbourhod = self.rotate_point_cloud(max_neighbourhod, pca_rot_3d)

            for k in self.Ks:
                accumulator = np.zeros((33, 33))
                accum_normals = np.zeros((k, 3))

                nearest_neighbour_point_indices = indices[point_index, :k]

                local_aniso_prob = aniso_probabilities[nearest_neighbour_point_indices]
                local_aniso_prob /= np.sum(local_aniso_prob)

                random_choices = np.random.choice(
                    k, (self.T, 3), replace=True, p=local_aniso_prob
                )

                for ia, ib, ic in random_choices:

                    if ia == ib or ib == ic or ia == ic:
                        [ia, ib, ic] = np.random.choice(
                            k, (3), replace=False, p=local_aniso_prob
                        )

                    [a, b, c] = point_cloud[[ia, ib, ic]]

                    normal = np.cross(b - a, c - a)
                    normal = normal / np.sum(normal)
                    if np.dot(normal, [0, 0, 1]) < 0:
                        # reorient normal
                        normal = normal * -1
