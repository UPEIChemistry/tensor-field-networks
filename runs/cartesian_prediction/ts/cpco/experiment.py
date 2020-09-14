from pathlib import Path
from tfn.tools.jobs import StructurePrediction


job = StructurePrediction(
    exp_config={
        "name": f"{Path(__file__).parent}",
        "notes": "",
        "seed": 1,
        "run_config": {"epochs": 1000, "test": False, "batch_size": 64},
        "loader_config": {
            "loader_type": "ts_loader",
            "splitting": "custom",
            "load_kwargs": {"remove_noise": True, "shuffle": False},
        },
        "builder_config": {
            "builder_type": "cartesian_builder",
            "prediction_type": "cartesians",
            "output_type": "cartesians",
        },
        "lr_config": {"min_delta": 0.01, "patience": 30, "cooldown": 20},
        "cm_config": {"write_rate": 2},
    }
)
job.run()
