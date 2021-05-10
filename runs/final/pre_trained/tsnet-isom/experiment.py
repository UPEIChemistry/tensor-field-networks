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
                    "loader_type": "isom_loader",
                    "path": "/home/riley/dev/python/tensor-field-networks/data/isomerization/isomerization_dataset.hd5f",
                    "splitting": "75:20:5",
                },
                "builder_config": {
                    "builder_type": "cartesian_builder",
                    "radial_factory": "multi_dense",
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
                    "radial_factory": "multi_dense",
                    "prediction_type": "cartesians",
                    "output_type": "cartesians",
                },
                "lr_config": {"min_delta": 0.01, "patience": 30, "cooldown": 20},
            }
        ),
    ],
)
job.run()
