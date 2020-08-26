from tfn.tools.jobs import Pipeline, Regression, StructurePrediction, CrossValidate


class TestPipeline:
    def test_regression_to_structure_prediction_to_cross_validation(
        self, builder_config, run_config
    ):
        job = Pipeline(
            jobs=[
                Regression(
                    exp_config={
                        "run_config": run_config,
                        "loader_config": {"loader_type": "iso17_loader"},
                        "builder_config": dict(
                            **builder_config, builder_type="force_builder"
                        ),
                    }
                ),
                StructurePrediction(
                    exp_config={
                        "run_config": run_config,
                        "loader_config": {
                            "loader_type": "qm9_loader",
                            "load_kwargs": {"modify_structures": True},
                        },
                        "builder_config": dict(
                            **builder_config, builder_type="cartesian_builder"
                        ),
                    }
                ),
                CrossValidate(
                    exp_config={
                        "run_config": run_config,
                        "loader_config": {"loader_type": "ts_loader", "splitting": 5},
                        "builder_config": dict(
                            **builder_config, builder_type="cartesian_builder"
                        ),
                    }
                ),
            ]
        )
        job.run()
