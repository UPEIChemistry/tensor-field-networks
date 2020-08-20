from tfn.tools.jobs import StructurePrediction

job = StructurePrediction(
    exp_config={
        "name": "DEFAULT TS MODEL ON TS DATASET",
        "notes": "Train on only distance matrix",
        "run_config": {"epochs": 200, "loss": "mae"},
        "builder_config": {"builder_type": "ts_builder", "num_layers": (2, 2, 2)},
        "loader_config": {
            "loader_type": "ts_loader",
            "load_kwargs": {"output_distance_matrix": True},
        },
    }
)
job.run()
