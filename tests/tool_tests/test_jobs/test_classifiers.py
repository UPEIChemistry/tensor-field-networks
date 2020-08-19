from tfn.tools.jobs import Regression


class TestClassifiers:
    def test_siamese_network(self, run_config, builder_config):
        loader_config = {
            "loader_type": "ts_loader",
            "load_kwargs": {"output_type": "siamese"},
        }
        job = Regression(
            {
                "name": "test",
                "run_config": run_config,
                "loader_config": loader_config,
                "builder_config": dict(
                    **builder_config, builder_type="siamese_builder"
                ),
            }
        )
        job.run()

    def test_basic_classifier(self, run_config, builder_config):
        loader_config = {
            "loader_type": "ts_loader",
            "load_kwargs": {"output_type": "classifier"},
        }
        job = Regression(
            {
                "name": "test",
                "run_config": run_config,
                "loader_config": loader_config,
                "builder_config": dict(
                    **builder_config, builder_type="classifier_builder"
                ),
            }
        )
        job.run()
