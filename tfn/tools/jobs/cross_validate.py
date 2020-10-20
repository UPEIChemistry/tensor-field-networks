from copy import copy

import numpy as np
from sacred.run import Run
from tensorflow.keras.models import Model

from . import KerasJob, config_defaults
from ..callbacks import CartesianMetrics


class CrossValidate(KerasJob):
    @property
    def config_defaults(self):
        base = super().config_defaults
        base["loader_config"][
            "map_points"
        ] = False  # Ensure reconstruction works properly
        base["cm_config"] = copy(config_defaults.cm_config)
        return base

    def _main(
        self,
        run: Run,
        seed: int,
        fitable: Model = None,
        fitable_config: dict = None,
        loader_config: dict = None,
    ):
        # folds: (((x1, x2, ...), (y1, y2, ...)), ...)
        model = None
        train_loss = []
        val_loss = []
        loader, folds = self._load_data(loader_config)
        print(f"**CROSS VALIDATING WITH {len(folds)} FOLDS**")
        root = self.exp_config["run_config"]["root_dir"]

        # Loop over folds
        for i in range(len(folds)):
            print(f"CROSS VALIDATING USING FOLD {i} AS VAL FOLD...")
            val = folds[i]
            train = self._combine_folds(folds[:i] + folds[i + 1 :])
            data = (train, val, None)  # No testing data
            model = self._load_fitable(loader, fitable_config)

            # Preload weights if necessary
            if fitable is not None:
                fitable.save_weights("./temp_weights.hdf5")
                model.load_weights("./temp_weights.hdf5")

            # fit the new model
            self.exp_config["run_config"]["root_dir"] = root / f"cv_model_{i}"
            model = self._fit(
                run,
                model,
                data,
                callbacks=[
                    CartesianMetrics(
                        self.exp_config["run_config"]["root_dir"] / "cartesians",
                        *data,
                        **self.exp_config["cm_config"],
                    )
                ],
            )

            # [(loss, metric1, metric2, ...), ...]
            train_loss.append(self._evaluate_fold(model, train))
            val_loss.append(self._evaluate_fold(model, val))

        loss = np.array([train_loss, val_loss])  # (2, num_folds, ?)
        print(f"AVERAGE TRAIN LOSS ACROSS MODELS {np.mean(loss[0], axis=0).tolist()}")
        print(f"STANDARD DEVIATION: {np.std(loss[0], axis=0).tolist()}")
        print("Final train losses: {}".format("\n".join(map(str, train_loss))))

        print(f"AVERAGE VAL LOSS ACROSS MODELS {np.mean(loss[1], axis=0).tolist()}")
        print(f"STANDARD DEVIATION: {np.std(loss[1], axis=0).tolist()}")
        print("Final val losses: {}".format("\n".join(map(str, val_loss))))
        return model

    def _evaluate_fold(self, fitable: Model, data: list):
        loss = fitable.evaluate(*data, verbose=0)
        if not isinstance(loss, list):
            loss = [loss]
        loss.append(
            CartesianMetrics.structure_loss(
                data[0][0], fitable.predict(data[0]), data[1][0]
            )[0]
        )
        return loss

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
