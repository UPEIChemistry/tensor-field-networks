from tfn.tools.jobs import Regression

job = Regression(
    exp_config={
        "name": "QM9 STRUCTURE PREDICTION USING CARTESIAN TS MODEL",
        "notes": "Using all defaults",
        "loader_config": {
            "loader_type": "qm9_loader",
            "load_kwargs": {"modify_structures": True, "output_distance_matrix": False},
        },
        "builder_config": {
            "builder_type": "ts_builder",
            "output_distance_matrix": False,
        },
    }
)
job.run()
