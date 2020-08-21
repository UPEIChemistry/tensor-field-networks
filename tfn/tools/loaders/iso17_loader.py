import re

import h5py
import numpy as np

from .data_loader import DataLoader


class ISO17DataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        self.use_energies = kwargs.pop("use_energies", True)
        self._force_mu = None
        self._force_sigma = None
        super().__init__(*args, **kwargs)

    @property
    def mu(self):
        atomic_means = np.array(
            [
                0.0,  # Dummy points
                -13.61312172,  # Hydrogens
                -1029.86312267,  # Carbons
                -2042.61123593,  # Oxygens
            ]
        ).reshape((-1, 1))
        if self._force_mu is None:
            self.load_data()
        return atomic_means, self._force_mu

    @property
    def sigma(self):
        if self._force_mu is None:
            self.load_data()
        return np.ones_like(self.mu[0]), self._force_sigma

    def load_data(self, *args, **kwargs):
        """
        The ISO17 has 640982 structures split across 5 different datasets. The ISO17 is composed of
        129 isomers of C7O2H10 each with 5000 structures (frames with 1 femtosecond resolution)
        along various MD trajectories. Each structure has an associated molecular energy,
        and set of atomic forces.

        :param kwargs: possible kwargs:
            return_stats: bool. Defaults to False. exit early by returning mu, sigma
            return_maxz: bool. Defaults to False. exit early by returning max_z
            dataset_type: str. Defaults to 'reference'. Indicator for which dataset to select for
            train/val split test_type: str. Defaults to 'test_other'. Indictator for which datset
            to select for testing
        :return: dict. The structure of the returned data is as such:
            {
                dataset: (x, y)
            }
            Possible values for dataset:
                'reference' - 80% of steps of 80% of MD trajectories, (404000 examples).
                'reference_eq' - equilibrium conformations of those molecules, (101 examples).
                'test_within' - remaining 20% unseen steps of reference trajectories,
                    (101000 examples).
                'test_other' - remaining 20% unseen MD trajectories, (130000 examples).
                'test_eq' - equilibrium conformations of test trajectories, (5881 examples).

        """
        if (
            self.data is not None
            and "dataset_type" not in kwargs
            and "test_type" not in kwargs
        ):
            return self.data
        dataset_name = kwargs.get("dataset_type", "reference")
        test_name = kwargs.get("test_type", "test_other")
        data = {
            dataset_name: [],  # Populated to [positions, atomic_nums, energies, forces]
            test_name: [],
        }

        # Load from hdf5 file
        with h5py.File(self.path, "r") as file:
            for name, l in data.items():
                positions = self.pad_along_axis(
                    np.array(file["{}/positions".format(name)]), self.num_points
                )
                atomic_nums = self.pad_along_axis(
                    np.tile(
                        np.expand_dims(file["{}/atomic_numbers".format(name)], axis=0),
                        (len(positions), 1),
                    ),
                    self.num_points,
                )

                energies = np.array(file["{}/energies".format(name)])
                forces = (
                    self.pad_along_axis(
                        np.array(file["{}/forces".format(name)]), self.num_points
                    )
                    * self.KCAL_PER_EV
                )

                if self.use_energies:
                    l.extend([atomic_nums, positions, energies, forces])
                else:
                    l.extend([atomic_nums, positions, forces])

        # Remapping
        if self.map_points:
            [self.remap_points(d[0]) for d in data.values()]
        self._max_z = np.max(data[dataset_name][0]) + 1
        if kwargs.get("return_maxz", False):
            return

        # Get Force mu/sigma
        self._force_mu = np.mean(data[dataset_name][-1])
        self._force_sigma = np.std(data[dataset_name][-1])

        # Split data
        self.splitting = re.search(r"\d{1,2}:\d{1,2}", self.splitting).group(0)
        self.data = self.three_way_split(
            data=[data[dataset_name][:2], data[dataset_name][2:]],
            length=len(data[dataset_name][0]),
        )
        self.splitting = None
        self.data.extend(
            self.three_way_split(
                data=[data[test_name][:2], data[test_name][2:]],
                length=len(data[dataset_name][0]),
            )
        )
        return self.data
