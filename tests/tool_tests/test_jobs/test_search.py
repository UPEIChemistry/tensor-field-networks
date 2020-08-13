import pytest

from tfn.tools.jobs import Search, GridSearch


@pytest.mark.skip(reason="Not testing kerastuner currently")
class TestRandomSearch:
    tuner_config = {"max_trials": 3}

    def test_defaults(
        self, run_config, factory_config, architecture_search, tuner_config
    ):
        job = Search(
            tuner="random",
            exp_config={
                "name": "test",
                "run_config": run_config,
                "tuner_config": dict(**self.tuner_config, **tuner_config),
                "factory_config": factory_config,
                "search_space": architecture_search,
            },
        )
        job.run()

    def test_radial_search(self, run_config, factory_config, tuner_config):
        job = Search(
            tuner="random",
            exp_config={
                "name": "test",
                "run_config": run_config,
                "tuner_config": dict(**self.tuner_config, **tuner_config),
                "factory_config": factory_config,
                "search_space": {
                    "radial_factory": {
                        "type": "choice",
                        "kwargs": {
                            "values": ["single_dense", "multi_conv", "single_conv"]
                        },
                    }
                },
            },
        )
        job.run()

    def test_dual_loss_weight_search(self, run_config, factory_config, tuner_config):
        factory_config["run_eagerly"] = False
        factory_config["dynamic"] = False
        self.tuner_config["project_name"] = "dual_loss_testing"
        job = Search(
            tuner="random",
            exp_config={
                "name": "test",
                "run_config": run_config,
                "tuner_config": dict(**self.tuner_config, **tuner_config),
                "loader_config": {"loader_type": "iso17_loader"},
                "factory_config": dict(**factory_config, factory_type="force_factory"),
                "search_space": {
                    "molecular_energy_weight": {
                        "type": "choice",
                        "kwargs": {"values": [0.01, 0.1, 0.25, 0.5, 1.0]},
                    }
                },
            },
        )
        job.run()


@pytest.mark.skip(reason="Not testing kerastuner currently")
class TestHyperbandSearch:
    tuner_config = {"max_epochs": 2}

    def test_defaults(
        self, run_config, factory_config, architecture_search, tuner_config
    ):
        factory_config["run_eagerly"] = False
        factory_config["dynamic"] = False
        self.tuner_config["project_name"] = "hyperband_testing"
        job = Search(
            tuner="hyperband",
            exp_config={
                "name": "test",
                "run_config": run_config,
                "tuner_config": dict(**self.tuner_config, **tuner_config),
                "factory_config": factory_config,
                "search_space": architecture_search,
            },
        )
        job.run()


class TestGridSearch:
    def test_basic_grid_search(self, run_config):
        grid_config = {
            "sum_points": [True, False],
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
