from tensorflow.keras.models import load_model

from tfn.tools.jobs import Regression, StructurePrediction


class TestScalarModels:
    def test_qm9(self, run_config, builder_config):
        job = Regression(
            {"name": "test", "run_config": run_config, "builder_config": builder_config}
        )
        job.run()

    def test_non_residual(self, run_config, builder_config):
        job = Regression(
            {
                "name": "test",
                "run_config": run_config,
                "builder_config": dict(**builder_config, residual=False),
            }
        )
        job.run()

    def test_sum_points(self, run_config, builder_config):
        job = Regression(
            {
                "name": "test",
                "run_config": run_config,
                "builder_config": dict(**builder_config, sum_points=True),
            }
        )
        job.run()

    def test_cosine_basis(self, run_config, builder_config):
        job = Regression(
            {
                "name": "test",
                "run_config": run_config,
                "builder_config": dict(**builder_config, basis_type="cosine"),
            }
        )
        job.run()

    def test_single_dense_radial(self, run_config, builder_config):
        job = Regression(
            {
                "name": "test",
                "run_config": run_config,
                "builder_config": dict(
                    **builder_config,
                    **{
                        "embedding_units": 32,
                        "model_num_layers": (3, 3, 3),
                        "si_units": 32,
                        "radial_factory": "single_dense",
                        "radial_kwargs": {
                            "num_layers": 1,
                            "units": 64,
                            "activation": "ssp",
                            "kernel_lambda": 0.01,
                            "bias_lambda": 0.01,
                        },
                    }
                ),
            }
        )
        job.run()

    def test_default_loads_graphly(self, run_config, builder_config, model):
        run_config["run_eagerly"] = False
        builder_config["dynamic"] = False
        job = Regression(
            {"name": "test", "run_config": run_config, "builder_config": builder_config}
        )
        job.run()
        model = load_model(model)
        assert True

    def test_default_loads_eagerly(self, run_config, builder_config, model):
        run_config["run_eagerly"] = True
        builder_config["dynamic"] = True
        job = Regression(
            {"name": "test", "run_config": run_config, "builder_config": builder_config}
        )
        job.run()
        model = load_model(model)
        assert True


class TestDualModels:
    def test_iso17(self, run_config, builder_config):
        loader_config = {"loader_type": "iso17_loader"}
        job = Regression(
            {
                "name": "test",
                "run_config": run_config,
                "loader_config": loader_config,
                "builder_config": dict(**builder_config, builder_type="force_builder"),
            }
        )
        job.run()

    def test_sn2(self, run_config, builder_config):
        loader_config = {"loader_type": "sn2_loader"}
        job = Regression(
            {
                "name": "test",
                "run_config": run_config,
                "loader_config": loader_config,
                "builder_config": dict(**builder_config, builder_type="force_builder"),
            }
        )
        job.run()


class TestCartesianModels:
    def test_cumulative_loss_cartesian_prediction_distance_matrix_output(
        self, run_config, builder_config
    ):
        loader_config = {
            "loader_type": "ts_loader",
            "load_kwargs": {"output_distance_matrix": True},
        }
        job = StructurePrediction(
            {
                "name": "test",
                "run_config": dict(
                    **run_config, loss="cumulative_loss", optimizer="sgd"
                ),
                "loader_config": loader_config,
                "builder_config": dict(
                    **builder_config,
                    builder_type="cartesian_builder",
                    prediction_type="cartesians",
                    output_type="distance_matrix"
                ),
            }
        )
        job.run()

    def test_vector_prediction_cartesian_output(self, run_config, builder_config):
        loader_config = {
            "loader_type": "ts_loader",
            "load_kwargs": {"output_distance_matrix": False},
        }
        job = StructurePrediction(
            {
                "name": "test",
                "run_config": run_config,
                "loader_config": loader_config,
                "builder_config": dict(
                    **builder_config,
                    builder_type="cartesian_builder",
                    prediction_type="vectors",
                    output_type="cartesians"
                ),
            }
        )
        job.run()

    def test_vector_prediction_distance_matrix_output(self, run_config, builder_config):
        loader_config = {
            "loader_type": "ts_loader",
            "load_kwargs": {"output_distance_matrix": True},
        }
        job = StructurePrediction(
            {
                "name": "test",
                "run_config": run_config,
                "loader_config": loader_config,
                "builder_config": dict(
                    **builder_config,
                    builder_type="cartesian_builder",
                    prediction_type="vectors",
                    output_type="distance_matrix"
                ),
            }
        )
        job.run()

    def test_cartesian_prediction_cartesian_output(self, run_config, builder_config):
        loader_config = {
            "loader_type": "ts_loader",
            "load_kwargs": {"output_distance_matrix": False},
        }
        job = StructurePrediction(
            {
                "name": "test",
                "run_config": run_config,
                "loader_config": loader_config,
                "builder_config": dict(
                    **builder_config,
                    builder_type="cartesian_builder",
                    prediction_type="cartesians",
                    output_type="cartesians"
                ),
            }
        )
        job.run()

    def test_cartesian_prediction_distance_matrix_output(
        self, run_config, builder_config
    ):
        loader_config = {
            "loader_type": "ts_loader",
            "load_kwargs": {"output_distance_matrix": True},
        }
        job = StructurePrediction(
            {
                "name": "test",
                "run_config": run_config,
                "loader_config": loader_config,
                "builder_config": dict(
                    **builder_config,
                    builder_type="cartesian_builder",
                    prediction_type="cartesians",
                    output_type="distance_matrix"
                ),
            }
        )
        job.run()

    def test_modified_qm9_vector_prediction_cartesian_output(
        self, run_config, builder_config
    ):
        loader_config = {
            "loader_type": "qm9_loader",
            "load_kwargs": {"modify_structures": True},
        }
        job = Regression(
            {
                "name": "test",
                "run_config": run_config,
                "loader_config": loader_config,
                "builder_config": dict(
                    **builder_config, builder_type="cartesian_builder"
                ),
            }
        )
        job.run()
