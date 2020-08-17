from tfn.tools.jobs import Pipeline


job = Pipeline(
    {
        "run_config": {
            "name": "tl0 - Modified QM9 Structure prediction to TS structure predictor",
            "notes": "Not freezing layers, defaults for both models",
        },
        "pipeline_config": {
            "configs": [
                {  # QM9 config
                    "builder_config": {"builder_type": "ts_builder"},
                    "loader_config": {
                        "loader_type": "qm9_loader",
                        "load_kwargs": {"modify_structures": True},
                    },
                },
                {  # TS config
                    "builder_config": {"builder_type": "ts_builder"},
                    "loader_config": {
                        "loader_type": "ts_loader",
                        "splitting": "75:20:5",
                        "load_kwargs": {"output_distance_matrix": True},
                    },
                },
            ]
        },
    }
)
job.run()
