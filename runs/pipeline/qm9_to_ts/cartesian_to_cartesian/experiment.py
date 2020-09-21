from pathlib import Path
from tfn.tools.jobs import Pipeline, StructurePrediction


job = Pipeline(
    exp_config={"name": f"{Path(__file__).parent}", "seed": 1},
    jobs=[
        StructurePrediction(
            exp_config={
                "name": f"{Path(__file__).parent} QM9",
                "seed": 1,
                "loader_config": {
                    "loader_type": "qm9_loader",
                    "load_kwargs": {"modify_structures": True},
                },
                "builder_config": {
                    "builder_type": "cartesian_builder",
                    "prediction_type": "cartesians",
                    "output_type": "cartesians",
                },
            }
        ),
        StructurePrediction(
            exp_config={
                "name": f"{Path(__file__).parent} TS",
                "seed": 1,
                "run_config": {"epochs": 1000, "test": False, "batch_size": 48},
                "loader_config": {
                    "loader_type": "ts_loader",
                    "splitting": "custom",
                    "map_points": False,
                    "load_kwargs": {"remove_noise": False, "shuffle": False},
                },
                "builder_config": {
                    "builder_type": "cartesian_builder",
                    "prediction_type": "cartesians",
                    "output_type": "cartesians",
                },
                "lr_config": {"min_delta": 0.01, "patience": 30, "cooldown": 20},
            }
        ),
    ],
)
job.run()
