from tfn.tools.jobs import CrossValidate


class TestCrossValidation:
    def test_cartesian_ts_cross_validated(self, run_config, builder_config):
        loader_config = {
            "loader_type": "ts_loader",
            "splitting": 5,
            "load_kwargs": {"output_distance_matrix": False},
        }
        job = CrossValidate(
            {
                "name": "test",
                "run_config": run_config,
                "loader_config": loader_config,
                "builder_config": dict(
                    **builder_config,
                    builder_type="ts_builder",
                    output_distance_matrix=False
                ),
            }
        )
        job.run()
