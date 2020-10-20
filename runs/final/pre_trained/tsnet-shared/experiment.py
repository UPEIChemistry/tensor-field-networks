from pathlib import Path
from tfn.tools.jobs import Pipeline, CrossValidate, StructurePrediction


job = Pipeline(
    exp_config={"name": f"{Path(__file__).parent}", "seed": 1},
    jobs=[
        StructurePrediction(
            exp_config={
                "name": f"{Path(__file__).parent} QM9",
                "seed": 1,
                "run_config": {"epochs": 50, "test": False,},
                "loader_config": {
                    "loader_type": "qm9_loader",
                    "splitting": "90:10:0",
                    "load_kwargs": {"modify_structures": True, "custom_maxz": 36},
                },
                "builder_config": {
                    "builder_type": "cartesian_builder",
                    "radial_factory": "single_dense",
                    "prediction_type": "cartesians",
                    "output_type": "cartesians",
                },
            }
        ),
        CrossValidate(
            exp_config={
                "name": f"{Path(__file__).parent} TS",
                "seed": 1,
                "run_config": {"epochs": 1000, "test": False, "batch_size": 48,},
                "loader_config": {
                    "loader_type": "ts_loader",
                    "splitting": 5,
                    "map_points": False,
                    "load_kwargs": {"remove_noise": True, "shuffle": False},
                },
                "builder_config": {
                    "builder_type": "cartesian_builder",
                    "radial_factory": "single_dense",
                    "prediction_type": "cartesians",
                    "output_type": "cartesians",
                },
                "lr_config": {"min_delta": 0.01, "patience": 30, "cooldown": 20},
            }
        ),
    ],
)
job.run()
