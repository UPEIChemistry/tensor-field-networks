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

        x: [atomic_numbers, reactant_cartesians, reactant_complex_cartesians,
            product_cartesians, product_complex_cartesians] if kwarg `input_type` ==
            'cartesians' (Default). If kwarg `input_type` == 'energies', then x is: [
            atomic_numbers, reactant_energies, product_energies]. Very rarely will this be
            useful, but the functionality exists.
        y: [ts_cartesians] if kwarg `output_type` == 'cartesians' (Default). If kwarg
        'output_type' == 'energies' then y is: [ts_energies], and if `output_type` == 'both'
        then y is: [ts_cartesians, ts_energies]

        :param kwargs: Possible kwargs:
            'cache': bool. Defaults to True.
            'input_type': str. Defaults to 'cartesians'. Possible values include ['cartesians',
                'energies', 'classifier'],
            'output_type': str. Defaults to 'cartesians'. Possible values include ['cartesians',
            'energies', 'both', 'classifier']
        :return: data in the format: [(x_train, y_train), (x_val, y_val), (x_test, y_test)]
        """
        if self._data is not None and kwargs.get('cache', True):
            return self._data

        # Load Data
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
            energies = {
                structure_type: np.asarray(dataset['{}/energies'.format(structure_type)])
                for structure_type in ('reactant', 'ts', 'product')
            }

        # Remap
        if self.map_atoms:
            self.remap_atoms(atomic_nums)
        self._max_z = np.max(atomic_nums) + 1
        if kwargs.get('return_maxz', False):
            return

        # Set up possible inputs/outputs
        x, y = [], []
        x_energies = [
            energies['reactant'],
            energies['product']
        ]
        x_cartesians = [
            atomic_nums,
            cartesians['reactant'],
            cartesians['reactant_complex'],
            cartesians['product'],
            cartesians['product_complex']
        ]
        y_energies = [
            energies['ts']
        ]
        y_cartesians = [
            cartesians['ts']
        ]

        # Select requested inputs/outputs
        input_type = kwargs.get('input_type', 'cartesians').lower()
        output_type = kwargs.get('output_type', 'cartesians').lower()
        if input_type == 'energies':
            x.extend(x_energies)
        elif input_type == 'both':
            x.extend(x_cartesians)
            x.extend(x_energies)
        else:
            x.extend(x_cartesians)
        if output_type == 'energies':
            y.extend(y_energies)
        elif output_type == 'both':
            y.extend(y_cartesians)
            y.extend(y_energies)
        else:
            y.extend(y_cartesians)

        # Classifier data is of a very diff. form
        if input_type == 'classifier' or output_type == 'classifier':
            tiled_atomic_nums = np.tile(atomic_nums, (5, 1))
            cartesians = np.concatenate(
                [a for a in cartesians.values()],
                axis=0)
            labels = np.zeros((len(tiled_atomic_nums),))
            labels[2 * len(atomic_nums): 3 * len(atomic_nums) + 1] = 1

            # Shuffle and set vars
            s = np.arange(len(labels))
            np.random.shuffle(s)
            x = [tiled_atomic_nums[s], cartesians[s]]
            y = [labels[s]]

        # Split and serve data
        self._data = self.split_dataset(
            data=[x, y],
            length=len(atomic_nums)
        )
        return self._data
