import re
from typing import Union

import numpy as np


class DataLoader(object):
    EV_PER_HARTREE = 27.2116
    KCAL_PER_HARTREE = 627.509
    KCAL_PER_EV = 23.06035

    def __init__(
        self,
        path: str,
        map_points: bool = True,
        splitting: Union[str, int, None] = "70:20:10",
        pre_load: bool = False,
        num_points: int = 29,
        **kwargs
    ):
        """

        :param path: str. Path to dataset, typically a .npz or .hdf5 file.
        :param map_points: bool. Defaults to True. Whether or not to map point integers
            (e.g. atomic numbers) into a smaller set, such that no integer is left unassigned.
            pass False if reconstructing .xyz files after model training.
        :param splitting: Union[str, int, None]. Defaults to '70:20:10'
        :param pre_load:
        :param num_points:
        :param kwargs:
        """
        self.path = path
        self.map_points = map_points
        self.splitting = splitting
        self.num_points = num_points
        self.data = None
        self.dataset_length = None

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
        if self.data is None or self.dataset_length is None:
            raise NotImplementedError(
                "Data and Length must be specified. Make sure a DataLoader "
                "subclass is being used."
            )
        if isinstance(self.splitting, int):
            return self.cross_validate()
        else:
            return self.three_way_split()

    def few_examples(self, num_examples: int = 5, **kwargs):
        data = self.load_data(**kwargs)
        truncated_data = []
        for split in data:  # [x, y]
            truncated_split = []
            for l in split:
                truncated_array = []
                for array in l:
                    truncated_array.append(array[:num_examples])
                truncated_split.append(truncated_array)
            truncated_data.append(truncated_split)
        return truncated_data

    def cross_validate(self, data: list = None, length: int = None):
        """
        :return: data in the format: [(x_0, y_0), (x_1, y_1), ...] for number of folds specified
            in splitting param.
        """
        data = data or self.data
        length = length or self.dataset_length
        if self.splitting < 2:
            raise ValueError(
                "Must provide a splitting param of 2 or more folds for cross "
                "validation"
            )
        fold_length, remainder = divmod(length, self.splitting)
        folds = [fold_length for _ in range(self.splitting)]
        if remainder > 0.25 * length:
            folds.append(remainder)
        else:
            folds[-1] += remainder
        return self.split_data(data, folds)

    def three_way_split(self, data: list = None, length: int = None):
        """
        :return: data in the format: [
            [
                (x_train, y_train),
                (x_val, y_val),
                (x_test, y_test)
            ]
        ]
        """
        data = data or self.data
        length = length or self.dataset_length
        if self.splitting is None:
            splits = [length]  # Use 100 percent of dataset as train data
        else:
            splits = [
                int(int(x) / 100 * length)
                for x in re.findall(r"(\d{1,2})", self.splitting)
            ]
            splits[int(np.argmax(splits))] += length - sum(
                splits
            )  # Add remainder to largest split
        return self.split_data(data, splits)

    @staticmethod
    def split_data(data: list, splits: list):
        """
        :param data: dataset in the form [x, y], where x and y are lists of ndarrays.
        :param splits: List[int].
        :return: data split according to splits, with None for splits with length == 0.
        """
        x_data, y_data = data
        output_data = []
        for i, split in enumerate(splits):
            cursor = sum(splits[:i])
            boundary = cursor + split
            output_data.append(
                (
                    [x[cursor:boundary] for x in x_data],
                    [y[cursor:boundary] for y in y_data],
                )
            )
        output_data = [o if len(o[0][0]) != 0 else None for o in output_data]
        return output_data

    @staticmethod
    def remap_points(atomic_nums):
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
        b = np.pad(array, pad_width=npad, mode="constant", constant_values=0)
        return b

    @staticmethod
    def shuffle_arrays(x, y, length):
        """
        :param x: list. input data to be shuffled.
        :param y: list. output data to be shuffled.
        :param length: int. number of examples in dataset.
        :return: List[list]. Input and output shuffled.
        """
        s = np.arange(length)
        np.random.shuffle(s)
        inp = [a[s] for a in x]
        out = [a[s] for a in y]
        return inp, out
