from sklearn.model_selection import ParameterGrid

from . import KerasJob


class GridSearch(KerasJob):
    def __init__(self, *args, **kwargs):
        self.total_models = kwargs.pop("total_models", None)
        super().__init__(*args, **kwargs)

    def _main(self, run, fitable=None, dataloader_config=None, fitable_config=None):
        config_grid = ParameterGrid(self.exp_config["grid_config"])
        for i, config in enumerate(config_grid):
            if (i + 1) == self.total_models:  # Stop when hitting max models
                break
            print("\nConfig set (not showing defaults): {}".format(config))
            [
                config.setdefault(k, v)
                for k, v in self.exp_config["builder_config"].items()
            ]
            self._new_model_path(i)
            try:
                super()._main(run, fitable_config=config)
            except Exception as e:
                print(
                    "Encountered exception in search, skipping configuration...\n Error "
                    "message: {}".format(e)
                )
                pass
