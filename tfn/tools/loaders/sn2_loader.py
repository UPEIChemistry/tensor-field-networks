import numpy as np

from . import ISO17DataLoader


class SN2Loader(ISO17DataLoader):
    @property
    def mu(self):
        atomic_means = np.array([
            0.,  # Dummy atoms
            -13.579407869766147,  # H
            -1028.9362774711024,  # C
            -2715.578463075019,  # F
            -12518.663203367176,  # Cl
            -70031.09203874589,  # Br
            -8096.587166328217,  # I
        ]).reshape([-1, 1]) * self.KCAL_PER_EV
        if self._force_mu is None:
            self.load_data()
        return atomic_means, self._force_mu

    def load_data(self, *args, **kwargs):
        if self._data is not None:
            return self._data

        # Load from .npz
        data = np.load(self.path)
        cartesians = self.pad_along_axis(data['R'], self.num_atoms)  # (mols, atoms, 3)
        atomic_nums = self.pad_along_axis(data['Z'], self.num_atoms)  # (mols, atoms)
        energies = data['E'] * self.KCAL_PER_EV  # (mols, )
        forces = self.pad_along_axis(data['F'], self.num_atoms) * self.KCAL_PER_EV
        dipoles = data['D']  # (mols, 3)

        # Remap atoms
        if self.map_atoms:
            self.remap_atoms(atomic_nums)
        self._max_z = np.max(atomic_nums) + 1
        if kwargs.get('return_maxz', False):
            return

        # Stats for forces/dipoles
        self._force_mu = np.mean(forces)
        self._force_sigma = np.std(forces)

        self._data = self.split_dataset(
            data=[
                [cartesians, atomic_nums],
                [energies, forces]
            ],
            length=len(atomic_nums)
        )
        return self._data
