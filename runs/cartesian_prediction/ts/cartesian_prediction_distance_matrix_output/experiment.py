from pathlib import Path
from tfn.tools.jobs import StructurePrediction


job = StructurePrediction(
    exp_config={
        "name": f"{Path(__file__).parent}",
        "notes": "",
        "seed": 608638837,
        "run_config": {"epochs": 500},
        "loader_config": {
            "loader_type": "ts_loader",
            "load_kwargs": {"output_distance_matrix": True},
        },
        "builder_config": {
            "builder_type": "cartesian_builder",
            "prediction_type": "cartesians",
            "output_type": "distance_matrix",
        },
        "lr_config": {"min_delta": 0.01, "patience": 30, "cooldown": 20},
    }
)
job.run()
