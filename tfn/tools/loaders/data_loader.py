import re
from typing import Union

import numpy as np


class DataLoader(object):
    EV_PER_HARTREE = 27.2116
    KCAL_PER_HARTREE = 627.509
    KCAL_PER_EV = 23.06035

    def __init__(self,
                 path: str,
                 map_atoms: bool = True,
                 splitting: Union[str, None] = '70:20:10',
                 pre_load: bool = False,
                 num_atoms: int = None,
                 **kwargs
                 ):
        self.path = path
        self.map_atoms = map_atoms
        self.splitting = splitting
        self.num_atoms = num_atoms

        self._data = None
        self._max_z = None
        self._mu = None
        self._sigma = None
        if pre_load:
            self.load_data()

    @property
    def max_z(self):
        if self._max_z is None:
            self.load_data(return_maxz=True)
        return self._max_z

    @property
    def mu(self):
        raise NotImplementedError

    @property
    def sigma(self):
        raise NotImplementedError

    def load_data(self, *args, **kwargs):
        """
        :return: data in the format: List[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray], ...]. e.g.
        [
            (x_train, y_train),
            (x_val, y_val),
            (x_test, y_test)
        ]
        """
        raise NotImplementedError

    def few_examples(self, num_examples: int = 5):
        train, *_ = self.load_data()  # train is [x, y] where x, y are lists with 1 or more item(s)
        sample = [[], []]
        for l, data in zip(sample, train):
            for d in data:
                l.append(d[:num_examples])
        return sample

    def split_dataset(self, data: list, length: int):
        """

        :param data: List[List[np.ndarray]]. Full x/y data in the format:
            [x, y], where x and y are lists of 1 or more ndarrays.
        :param length: int. Total number of training examples.
        :return: data in the format: [
            [
                (x_train, y_train),
                (x_val, y_val),
                (x_test, y_test)
            ]
        ]
        """
        x_data, y_data = data
        if self.splitting is None:
            splits = [length]  # Use 100 percent of dataset as train data
        else:
            splits = [
                int(int(x) / 100 * length)
                for x in re.findall(r'(\d{1,2})', self.splitting)
            ]
        output_data = []
        for i in range(len(splits)):
            if i == 0:
                first_split = splits[0]
                output_data.append(
                    (
                        [x[:first_split] for x in x_data],
                        [y[:first_split] for y in y_data]
                    )
                )
            elif i == 1:
                first_split, second_split = splits[:2]
                output_data.append(
                    (
                        [x[first_split:first_split + second_split] for x in x_data],
                        [y[first_split:first_split + second_split] for y in y_data],
                    )
                )
            elif i == 2:
                second_split = splits[1]
                output_data.append(
                    (
                        [x[second_split:] for x in x_data],
                        [y[second_split:] for y in y_data]
                    )
                )

        return output_data

    @staticmethod
    def remap_atoms(atomic_nums):
        atom_mapping = np.unique(atomic_nums)
        for remapped_z, original_z in enumerate(atom_mapping):
            atomic_nums[atomic_nums == original_z] = remapped_z

    @staticmethod
    def pad_along_axis(array: np.ndarray, target_length, axis=1):
        pad_size = target_length - array.shape[axis]
        axis_nb = len(array.shape)
        if pad_size < 0:
            return array
        npad = [(0, 0) for _ in range(axis_nb)]
        npad[axis] = (0, pad_size)
        b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)
        return b
