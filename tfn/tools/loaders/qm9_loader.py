import h5py
import numpy as np

from tfn.layers.atomic_images import OneHot

from ...layers.utility_layers import MaskedDistanceMatrix
from .data_loader import DataLoader


class QM9DataDataLoader(DataLoader):
    @property
    def mu(self):
        a = np.zeros((self.max_z, 1), dtype="float32")
        a[1] = -13.61312172
        a[6] = -1029.86312267
        a[7] = -1485.30251237
        a[8] = -2042.61123593
        a[9] = -2713.48485589
        return a

    @property
    def sigma(self):
        return np.ones_like(self.mu)

    def load_data(self, *args, **kwargs):
        """
        The QM9 is a dataset of 133,885 small molecules with up to 9 heavy points (C, N, O, or F). The dataset has 13
        different chemical properties associated with each structure, the most valuable being energy, of which there
        are several forms. This `DataLoader` is responsible for returning U0 energies; the internal energy of the
        structures at 298 Kelvin.

        :param kwargs: possible kwargs:
            return_stats: exit early by returning mu, sigma
            return_maxz: exit early by returning max_z
        :return: List[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray], ...].
            QM9 data in the format:
            [
                (x_train, y_train),
                (x_val, y_val),
                (x_test, y_test)
            ], where x = (cartesians, atomic_nums) and y = energies
        """
        if self.data is not None:
            return self.data
        with h5py.File(self.path, "r") as dataset:
            cartesians = self.pad_along_axis(
                np.nan_to_num(dataset["QM9/R"]), self.num_points
            )
            atomic_nums = self.pad_along_axis(
                np.array(dataset["QM9/Z"]), self.num_points
            )
            energies = np.array(dataset["QM9/U_naught"]) * self.EV_PER_HARTREE

        if self.map_points:
            self.remap_points(atomic_nums)
        self._max_z = kwargs.get("custom_maxz", None) or np.max(atomic_nums) + 1
        if kwargs.get("return_maxz", False):
            return

        if kwargs.get("modify_structures", False):
            forward_cartesians, reverse_cartesians = self.modify_structures(
                cartesians,
                kwargs.get("modify_distance", 0.5),
                kwargs.get("modify_seed", 0),
            )
            if kwargs.get("classifier_output", False):
                tiled_cartesians = np.concatenate(
                    [cartesians, forward_cartesians, reverse_cartesians], axis=0
                )
                tiled_atomic_nums = np.tile(atomic_nums, (3, 1))
                labels = np.zeros((len(tiled_cartesians),), dtype="int32")
                labels[: len(cartesians)] = 1
                x, y = self.shuffle_arrays(
                    [tiled_atomic_nums, tiled_cartesians], [labels], len(labels)
                )
                length = len(labels)
            else:
                x = [atomic_nums, forward_cartesians, reverse_cartesians]
                y = [
                    np.triu(
                        MaskedDistanceMatrix()(
                            [OneHot(self.max_z)(atomic_nums), cartesians]
                        )
                    )
                    if kwargs.get("output_distance_matrix", False)
                    else cartesians
                ]
                length = len(atomic_nums)

        else:
            x = [atomic_nums, cartesians]
            y = [energies]
            length = len(atomic_nums)

        self.data = [x, y]
        self.dataset_length = length
        return super().load_data(*args, **kwargs)

    def modify_structures(self, c, distance=0.75, seed=0):
        np.random.seed(seed)
        indices = np.random.randint(3, size=(len(c)))
        forward, reverse = np.copy(c), np.copy(c)
        forward += 0.2 * distance
        reverse -= 0.2 * distance
        for i, j in enumerate(indices):
            forward[i, j] += 0.8 * distance
            reverse[i, j] -= 0.8 * distance
        return forward, reverse
