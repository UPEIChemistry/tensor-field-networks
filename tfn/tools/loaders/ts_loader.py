from h5py import File
import numpy as np

from . import DataLoader


class TSLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        self.use_energies = kwargs.pop('use_energies', False)
        kwargs.setdefault('num_atoms', 29)
        super().__init__(*args, **kwargs)

    @property
    def mu(self):
        if self.use_energies:
            return np.array(
                [
                    0.,  # Dummy atoms
                    -13.61312172,  # Hydrogens
                    -1029.86312267,  # Carbons
                    -1485.30251237,  # Nitrogens
                    -2042.61123593,  # Oxygens
                    -2715.57846308,  # Fluorines
                    -17497.9266683,  # Silicon
                    -19674.5108670,  # Phosphorus
                    -10831.2647155,  # Sulfur
                    -12518.6632034,  # Chlorine
                    -61029.6106422,  # Selenium
                    -70031.0920387,  # Bromine
                ]
            ).reshape((-1, 1))
        else:
            return 0

    @property
    def sigma(self):
        if self.use_energies:
            return np.ones_like(self.mu)
        else:
            return 1

    def load_data(self, *args, **kwargs):
        """
        The MP2 TS Dataset is a dataset of 74 gas-phase SN2 structures, comprised of
        reactant/ts/product cartesians & atomic numbers obtained using the MP2 method with the
        cc-PVDZ basis set.

        :return: data in the format:
        [
            (x_train, y_train),
            (x_val, y_val),
            (x_test, y_test)
        ]
        Where x is: [atomic_numbers, reactant_cartesians, reactant_complex_cartesians,
        product_cartesians, product_complex_cartesians]
        Where y is: [ts_cartesians]
        """
        if self._data is not None:
            return self._data
        with File(self.path, 'r') as dataset:
            atomic_nums = self.pad_along_axis(
                np.asarray(dataset['ts/atomic_numbers'], dtype='int'),
                self.num_atoms)
            cartesians = {
                structure_type: self.pad_along_axis(
                    np.nan_to_num(dataset['{}/cartesians'.format(structure_type)]),
                    self.num_atoms
                ) for structure_type in
                ('reactant', 'reactant_complex', 'ts', 'product_complex', 'product')
            }

        if self.map_atoms:
            self.remap_atoms(atomic_nums)
        self._max_z = np.max(atomic_nums) + 1
        if kwargs.get('return_maxz', False):
            return

        x = [
            atomic_nums,
            cartesians['reactant'],
            cartesians['reactant_complex'],
            cartesians['product'],
            cartesians['product_complex']
        ]
        y = [
            cartesians['ts']
        ]
        self._data = self.split_dataset(
            data=[x, y],
            length=len(atomic_nums)
        )
        return self._data

    def load_distance_data(self, *args, **kwargs):
        """DEPRICATED"""
        with File(self.path, 'r') as dataset:
            x = np.array(dataset['equilibria_distances'])
            y = np.array(dataset['ts_distances'])

        x = np.concatenate(
            [x, np.concatenate([x[:, :2], x[:, 3:4], x[:, 2:3], x[:, 5:], x[:, 4:5]], axis=-1)]
        )
        y = np.concatenate(
            [y, np.concatenate([y[:, 1:], y[:, :1]], axis=-1)]
        )

        if kwargs.get('split', False):
            (x_train, y_train), (x_test, y_test) = self.split_dataset([[x], [y]], length=len(x))
            return (x_train[0], y_train[0]), (x_test[0], y_test[0])
        else:
            return x, y
