import h5py
import numpy as np
from atomic_images.np_layers import DistanceMatrix

from .data_loader import DataLoader


class QM9DataDataLoader(DataLoader):
    def __init__(self,
                 *args,
                 **kwargs):
        kwargs.setdefault('num_atoms', 29)
        super().__init__(*args, **kwargs)

    @property
    def mu(self):
        return np.array(
            [
                0.,  # Dummy atoms
                -13.61312172,  # Hydrogens
                -1029.86312267,  # Carbons
                -1485.30251237,  # Nitrogens
                -2042.61123593,  # Oxygens
                -2713.48485589  # Fluorines
            ]
        ).reshape((-1, 1))  * self.KCAL_PER_EV

    @property
    def sigma(self):
        return np.ones_like(self.mu)

    def load_data(self, *args, **kwargs):
        """
        The QM9 is a dataset of 133,885 small molecules with up to 9 heavy atoms (C, N, O, or F). The dataset has 13
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
        if self._data is not None:
            return self._data
        with h5py.File(self.path, 'r') as dataset:
            cartesians = self.pad_along_axis(np.nan_to_num(dataset['QM9/R']), self.num_atoms)
            atomic_nums = self.pad_along_axis(np.array(dataset['QM9/Z']), self.num_atoms)
            energies = np.array(dataset['QM9/U_naught']) * self.KCAL_PER_HARTREE

        if self.map_atoms:
            self.remap_atoms(atomic_nums)
        self._max_z = np.max(atomic_nums) + 1
        if kwargs.get('return_maxz', False):
            return

        if kwargs.get('modify_structures', False):
            forward_cartesians, reverse_cartesians = self.modify_structures(
                cartesians,
                kwargs.get('modify_distance', 0.5),
                kwargs.get('modify_seed', 0))
            x = [atomic_nums, forward_cartesians, reverse_cartesians]
            y = [DistanceMatrix()(cartesians)]
        else:
            x = [cartesians, atomic_nums]
            y = [energies]
        self._data = self.split_dataset([x, y], len(atomic_nums))
        return self._data

    def modify_structures(self, c, distance=0.5, seed=0):
        np.random.seed(seed)
        indices = np.random.randint(3, size=(len(c)))
        forward, reverse = np.copy(c), np.copy(c)
        forward += (0.1 * distance)
        reverse -= (0.1 * distance)
        for i, j in enumerate(indices):
            forward[i, j] += (0.9 * distance)
            reverse[i, j] -= (0.9 * distance)
        return forward, reverse
