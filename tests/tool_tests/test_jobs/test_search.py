from tfn.tools.jobs import CrossValidate, GridSearch, Regression


class TestGridSearch:
    GRID_CONFIG = {
        "sum_atoms": [True, False],
        "model_num_layers": [(1, 1), (1, 1, 1)],
        "radial_kwargs": [
            {
                "num_layers": 2,
                "units": 18,
                "activation": "ssp",
                "kernel_lambda": 0.01,
                "bias_lambda": 0.01,
            },
            {
                "num_layers": 1,
                "units": 32,
                "activation": "ssp",
                "kernel_lambda": 0.01,
                "bias_lambda": 0.01,
            },
        ],
    }

    def test_basic_grid_search(self, run_config):
        job = GridSearch(
            job=Regression(exp_config={"name": "test", "run_config": run_config}),
            grid=self.GRID_CONFIG,
            total_models=3,
        )
        job.run()

    def test_cross_validate_grid_search(self, run_config, builder_config):
        loader_config = {
            "loader_type": "ts_loader",
            "splitting": 3,
            "load_kwargs": {"output_distance_matrix": True},
        }
        job = GridSearch(
            job=CrossValidate(
                {
                    "name": "test",
                    "run_config": dict(**run_config),
                    "loader_config": loader_config,
                    "builder_config": dict(
                        **builder_config,
                        builder_type="cartesian_builder",
                        prediction_type="vectors",
                        output_type="distance_matrix"
                    ),
                }
            ),
            grid=self.GRID_CONFIG,
            total_models=3,
        )
        job.run()
