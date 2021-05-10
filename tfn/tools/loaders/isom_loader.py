import numpy as np
import h5py
from . import DataLoader


class IsomLoader(DataLoader):
    @property
    def mu(self):
        return 0

    @property
    def sigma(self):
        return 1

    def load_data(self, *args, **kwargs):
        with h5py.File(self.path, "r") as dataset:
            atomic_nums = self.pad_along_axis(
                np.asarray(dataset["ts_train/atomic_nums"], dtype="int"),
                self.num_points,
            )
            cartesians = {
                structure_type: self.pad_along_axis(
                    np.nan_to_num(dataset["{}/cartesians".format(structure_type)]),
                    self.num_points,
                )
                for structure_type in ("ts_train", "r_train", "p_train",)
            }

        # Remap points
        if self.map_points:
            self.remap_points(atomic_nums)
        self._max_z = kwargs.get("custom_maxz", None) or np.max(atomic_nums) + 1
        if kwargs.get("return_maxz", False):
            return

        x = [atomic_nums, cartesians["r_train"], cartesians["p_train"]]
        y = [cartesians["ts_train"]]
        self.data = [x, y]
        self.dataset_length = len(atomic_nums)
        return super().load_data()
