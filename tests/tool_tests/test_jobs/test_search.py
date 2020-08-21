from tfn.tools.jobs import GridSearch


class TestGridSearch:
    def test_basic_grid_search(self, run_config):
        grid_config = {
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
        job = GridSearch(
            total_models=3,
            exp_config={
                "name": "test",
                "run_config": run_config,
                "grid_config": grid_config,
            },
        )
        job.run()
