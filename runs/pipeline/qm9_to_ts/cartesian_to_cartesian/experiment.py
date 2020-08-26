from pathlib import Path
from tfn.tools.jobs import Pipeline, StructurePrediction, CrossValidate


job = Pipeline(
    exp_config={"name": f"{Path(__file__).parent}", "seed": 608638837,},
    jobs=[
        StructurePrediction(
            exp_config={
                "name": f"{Path(__file__).parent}",
                "notes": "",
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
        CrossValidate(
            exp_config={
                "name": f"{Path(__file__).parent}",
                "notes": "",
                "seed": 608638837,
                "run_config": {
                    "epochs": 500,
                    "loss": "mae",
                    "metrics": ["cumulative_loss"],
                },
                "loader_config": {"loader_type": "ts_loader"},
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
