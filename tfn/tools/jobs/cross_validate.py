import numpy as np
from sacred.run import Run
from tensorflow.keras.models import Model

from . import KerasJob


class CrossValidate(KerasJob):
    def _main(
        self,
        run: Run,
        fitable: Model = None,
        fitable_config: dict = None,
        loader_config: dict = None,
    ):
        # folds: (((x1, x2, ...), (y1, y2, ...)), ...)
        loader, folds = self._load_data(loader_config)
        fitable = fitable or self._load_fitable(loader, fitable_config)
        train_loss = []
        val_loss = []
        print(f"**CROSS VALIDATING WITH {len(folds)} FOLDS**")
        for i in range(len(folds)):
            print(f"CROSS VALIDATING USING FOLD {i} AS VAL FOLD...")
            val = folds[i]
            train = self._combine_folds(folds[:i] + folds[i + 1 :])
            data = (train, val)
            self.exp_config["run_config"]["fit_verbosity"] = 0  # keep fitting quiet
            fitable = self._fit(run, fitable, data)
            train_loss.append(fitable.evaluate(*train))
            val_loss.append(fitable.evaluate(*val))
            if self.exp_config["run_config"]["save_model"]:
                self.exp_config["run_config"][
                    "save_verbosity"
                ] = 0  # avoid printing model arch.
                self._new_model_path(i)
                self._save_fitable(run, fitable)
        print(f"AVERAGE TRAIN LOSS ACROSS MODELS {np.mean(train_loss)}")
        print(f"AVERAGE VAL LOS ACROSS MODELS {np.mean(val_loss)}")
        return fitable

    @staticmethod
    def _combine_folds(folds):
        """
        :param folds: list. Folds to be combined. Of shape ((x, y), ...) where x and y are lists
            of ndarrays
        :return: list. Folds concatenated to the shape (x, y), where x and y are lists of
            ndarrays concatenated along axis 0 across all folds.
        """
        x_arrays = [[] for _ in folds[0][0]]
        y_arrays = [[] for _ in folds[0][1]]
        for (x, y) in folds:
            for j, array in enumerate(x):
                x_arrays[j].append(array)
            for j, array in enumerate(y):
                y_arrays[j].append(array)
        combined_folds = [
            [np.concatenate(x, axis=0) for x in x_arrays],
            [np.concatenate(y, axis=0) for y in y_arrays],
        ]
        return combined_folds
