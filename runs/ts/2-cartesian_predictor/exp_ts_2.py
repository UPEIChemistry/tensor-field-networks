from tfn.tools.jobs import SingleModel

job = SingleModel(
    exp_config={
        "name": "CARTESIAN TS MODEL ON TS DATASET",
        "notes": "Train on only distance matrix",
        "seed": 0,
        "run_config": {"epochs": 300, "loss": "mae", "write_test_results": True},
        "builder_config": {
            "builder_type": "ts_builder",
            "num_layers": (2, 2, 2),
            "output_distance_matrix": False,
        },
        "loader_config": {
            "loader_type": "ts_loader",
            "splitting": "75:15:10",
            "map_points": False,
            "load_kwargs": {"output_distance_matrix": False},
        },
    }
)
job.run()
