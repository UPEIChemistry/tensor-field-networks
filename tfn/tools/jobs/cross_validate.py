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
        loader, folds = self._load_data(loader_config)  # folds: ((x, y), (x, y), ...)
        fitable = fitable or self._load_fitable(loader, fitable_config)
        train_loss = []
        val_loss = []
        print(f"**CROSS VALIDATING WITH {len(folds)} FOLDS**")
        for i in range(len(folds)):
            val = folds[i]
            # TODO: Need to loop over folds, summing each array of x, y seperately across the folds
            # TODO: Ultimately, we need to turn folds: ((x, y), ...) into (x, y).
            # TODO: Probably need to isolate list of each distinct array in x and y for a
            # TODO: np.concatenate call, but not sure how to do that.
            train = []
            data = (train, val)
            fitable = self._fit(run, fitable, data)
            train_loss.append(fitable.evaluate(*train))
            val_loss.append(fitable.evaluate(*val))
            if self.exp_config["run_config"]["save_model"]:
                self._save_fitable(run, fitable)
            return fitable
