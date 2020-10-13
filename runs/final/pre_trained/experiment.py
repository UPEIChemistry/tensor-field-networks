import os
from pathlib import Path
from tfn.tools.jobs import Pipeline, StructurePrediction, LoadModel, CrossValidate

os.environ["DATADIR"] = "/home/riley/dev/python/data"

job = Pipeline(
    exp_config={"name": f"{Path(__file__).parent}", "seed": 1},
    jobs=[
        LoadModel(
            exp_config={
                "name": f"{Path(__file__).parent} QM9",
                "seed": 1,
                "run_config": {"model_path": "./trained_qm9_model.h5"},
            }
        ),
        StructurePrediction(
            exp_config={
                "name": f"{Path(__file__).parent} TS",
                "seed": 1,
                "run_config": {"epochs": 1000, "test": False, "batch_size": 48,},
                "loader_config": {
                    "loader_type": "ts_loader",
                    "splitting": "custom",
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
                "cm_config": {"write_rate": 50},
            }
        ),
    ],
)
job.run()
