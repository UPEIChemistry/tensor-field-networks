from pathlib import Path
from tfn.tools.jobs import CrossValidate


job = CrossValidate(
    exp_config={
        "name": f"{Path(__file__).parent}",
        "notes": "",
        "seed": 1,
        "run_config": {"epochs": 1000, "test": False, "batch_size": 48},
        "loader_config": {
            "path": "../../../../data/ts.hdf5"
            "loader_type": "ts_loader",
            "splitting": 5,
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
)
job.run()
