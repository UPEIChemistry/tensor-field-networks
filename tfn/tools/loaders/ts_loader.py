from h5py import File
import numpy as np

from tfn.layers.atomic_images import OneHot

from ...layers.utility_layers import MaskedDistanceMatrix
from . import DataLoader


class TSLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        self.use_energies = kwargs.pop("use_energies", False)
        super().__init__(*args, **kwargs)

    @property
    def mu(self):
        mu = np.array(
            [
                0.0,  # Dummy points
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
        if self.use_energies:
            return mu
        else:
            return np.zeros_like(mu)

    @property
    def sigma(self):
        return np.ones_like(self.mu)

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
                'classifier', 'siamese'],
            'output_type': str. Defaults to 'cartesians'. Possible values include ['cartesians',
            'energies', 'both', 'classifier', 'siamese']
        :return: data in the format: [(x_train, y_train), (x_val, y_val), (x_test, y_test)]
        """
        if (
            self.data is not None
            and self.dataset_length is not None
            and kwargs.get("cache", True)
        ):
            return super().load_data()

        # Load Data
        with File(self.path, "r") as dataset:
            atomic_nums = self.pad_along_axis(
                np.asarray(dataset["ts/atomic_numbers"], dtype="int"), self.num_points
            )
            cartesians = {
                structure_type: self.pad_along_axis(
                    np.nan_to_num(dataset["{}/cartesians".format(structure_type)]),
                    self.num_points,
                )
                for structure_type in (
                    "ts",
                    "reactant",
                    "reactant_complex",
                    "product_complex",
                    "product",
                )
            }
            energies = {
                structure_type: np.asarray(
                    dataset["{}/energies".format(structure_type)]
                )
                * self.EV_PER_HARTREE
                for structure_type in ("ts", "reactant", "product")
            }
            noisy_indices = np.asarray(
                dataset["noisy_reactions"], dtype="int"
            )  # (16, )

        # Pull out noise
        if kwargs.get("remove_noise", False):
            atomic_nums = np.delete(atomic_nums, noisy_indices, axis=0)
            cartesians = {
                k: np.delete(c, noisy_indices, axis=0) for k, c in cartesians.items()
            }
            energies = {
                k: np.delete(e, noisy_indices, axis=0) for k, e in energies.items()
            }

        # Remap
        if self.map_points:
            self.remap_points(atomic_nums)
        self._max_z = np.max(atomic_nums) + 1
        if kwargs.get("return_maxz", False):
            return

        # Determine I/O data
        input_type = kwargs.get("input_type", "cartesians").lower()
        output_type = kwargs.get("output_type", "cartesians").lower()

        if input_type == "classifier" or output_type == "classifier":
            tiled_atomic_nums, tiled_cartesians, labels = self.tile_arrays(
                atomic_nums, cartesians
            )
            x, y = self.shuffle_arrays(
                [tiled_atomic_nums, tiled_cartesians], [labels], len(labels)
            )
            length = len(labels)

        elif input_type == "siamese" or output_type == "siamese":
            x, y = self.make_siamese_dataset(
                *self.tile_arrays(
                    atomic_nums, cartesians, blacklist=kwargs.pop("blacklist", None)
                )
            )
            if kwargs.get("shuffle", True):
                x, y = self.shuffle_arrays(x, y, len(y[0]))
            length = len(y[0])

        else:  # Regression dataset
            length = len(atomic_nums)
            x = [
                atomic_nums,
                cartesians["reactant_complex"]
                if kwargs.get("use_complexes", False)
                else cartesians["reactant"],
                cartesians["product_complex"]
                if kwargs.get("use_complexes", False)
                else cartesians["product"],
            ]
            y = [
                np.triu(
                    MaskedDistanceMatrix()(
                        [OneHot(self.max_z)(atomic_nums), cartesians["ts"]]
                    )
                )
                if kwargs.get("output_distance_matrix", False)
                else cartesians["ts"],
                energies["ts"],
            ]
            if output_type == "energies":
                y.pop(0)
            elif output_type == "both":
                pass
            else:
                y.pop(1)

            # shuffle dataset
            if kwargs.get("shuffle", True):
                x, y = self.shuffle_arrays(x, y, length)

        # Split and serve data
        self.data = [x, y]
        self.dataset_length = length
        if self.splitting == "custom":
            split = [
                0,  # hetero-ring structure, complex
                3,  # 3 member double bond ring, simple reaction
                7,  # methyl-chloride, super simple
                11,  # ?
                16,  # ispropyl-chloride, little more complex
                22,  # ?
                24,  # Triple bond, perfect midpoint
            ]
            val = [[a[split] for a in x], [a[split] for a in y]]
            train = [
                [np.delete(a, split, 0) for a in x],
                [np.delete(a, split, 0) for a in y],
            ]
            return train, val, None
        else:
            return super().load_data()

    def make_siamese_dataset(self, tiled_atomic_nums, tiled_cartesians, labels):
        # Make x shape: (batch, batch, 2, points, 3) Convert for output -> (batch, 2, points, 3)
        c = np.zeros((len(labels), len(labels), 2, self.num_points, 3))
        a = np.zeros(c.shape[:-1])
        diff = np.where(
            (np.expand_dims(labels, -1) - np.expand_dims(labels, -2)) != 0, 1, 0
        )
        indices = np.triu_indices(diff.shape[0], 1)
        for i, (i_atomic_nums, i_cartesians) in enumerate(
            zip(tiled_atomic_nums, tiled_cartesians)
        ):
            for j, (j_atomic_nums, j_cartesians) in enumerate(
                zip(tiled_atomic_nums, tiled_cartesians)
            ):
                a[i, j, 1], c[i, j, 1] = i_atomic_nums, i_cartesians
                a[i, j, 0], c[i, j, 0] = j_atomic_nums, j_cartesians

        # assign data
        labels = [diff[indices]]
        x = [a[indices], c[indices]]
        return x, labels

    @staticmethod
    def tile_arrays(atomic_nums, cartesians, blacklist: list = None):
        """:return: tiled/concatenated arrays: [atomic_nums, cartesians <- (concat), labels]"""
        blacklist = blacklist or []
        tiled_atomic_nums = np.tile(atomic_nums, (5 - len(blacklist), 1))
        tiled_cartesians = np.concatenate(
            [a for key, a in cartesians.items() if key not in blacklist], axis=0
        )
        labels = np.zeros((len(tiled_atomic_nums),), dtype="int32")
        labels[: len(atomic_nums)] = 1
        return tiled_atomic_nums, tiled_cartesians, labels
