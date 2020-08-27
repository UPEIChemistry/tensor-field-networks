from pathlib import Path
from tfn.tools.jobs import StructurePrediction, CrossValidate


job = StructurePrediction(
    exp_config={
        "name": f"{Path(__file__).parent}",
        "notes": "",
        "seed": 0,
        "run_config": {
            "epochs": 500,
            "loss": "mae",
            "metrics": ["cumulative_loss"],
            "test": False,
        },
        "loader_config": {
            "loader_type": "ts_loader",
            "splitting": "90:10:0",
            "load_kwargs": {"remove_noise": True},
        },
        "builder_config": {
            "builder_type": "cartesian_builder",
            "prediction_type": "cartesians",
            "output_type": "cartesians",
        },
        "lr_config": {"min_delta": 0.01, "patience": 30, "cooldown": 20},
    }
)
job.run()
