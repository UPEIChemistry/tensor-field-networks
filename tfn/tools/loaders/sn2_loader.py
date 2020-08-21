import numpy as np

from . import DataLoader


class SN2Loader(DataLoader):
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
                -13.579407869766147,  # H
                -1028.9362774711024,  # C
                -2715.578463075019,  # F
                -12518.663203367176,  # Cl
                -70031.09203874589,  # Br
                -8096.587166328217,  # I
            ]
        ).reshape([-1, 1])
        if self._force_mu is None:
            self.load_data()
        return atomic_means, self._force_mu

    @property
    def sigma(self):
        if self._force_mu is None:
            self.load_data()
        return np.ones_like(self.mu[0]), self._force_sigma

    def load_data(self, *args, **kwargs):
        if self.data is not None:
            return self.data

        # Load from .npz
        data = np.load(self.path)
        cartesians = self.pad_along_axis(
            data["R"], self.num_points
        )  # (batch, points, 3)
        atomic_nums = self.pad_along_axis(data["Z"], self.num_points)  # (batch, points)
        energies = data["E"]  # (batch, )
        forces = self.pad_along_axis(data["F"], self.num_points)
        dipoles = data["D"]  # (batch, 3)

        # Remap points
        if self.map_points:
            self.remap_points(atomic_nums)
        self._max_z = np.max(atomic_nums) + 1
        if kwargs.get("return_maxz", False):
            return

        # Stats for forces/dipoles
        self._force_mu = np.mean(forces)
        self._force_sigma = np.std(forces)

        self.data = [[atomic_nums, cartesians], [energies, forces]]
        self.dataset_length = len(atomic_nums)
        return super().load_data()
