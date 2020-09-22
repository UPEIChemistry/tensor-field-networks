from pathlib import Path
from tfn.tools.jobs import GridSearch, StructurePrediction
from tfn.tools.jobs.config_defaults import default_grid_search


job = GridSearch(
    job=StructurePrediction(
        exp_config={
            "name": f"{Path(__file__).parent}",
            "notes": "~1000 models total",
            "seed": 1,
            "run_config": {
                "epochs": 1000,
                "test": False,
                "save_model": False,
                "fit_verbosity": 0,
                "batch_size": 48,
                "use_strategy": True,
                "use_default_callbacks": False,
            },
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
            "cm_config": {"write_rate": 0, "max_structures": 0},
        }
    ),
    grid=default_grid_search,
)
job.run()
