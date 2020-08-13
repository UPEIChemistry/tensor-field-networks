from tfn.tools.jobs import GridSearch
from tfn.tools.jobs.config_defaults import default_grid_search

job = GridSearch(
    total_models=100,
    exp_config={
        "name": "TS Siamese architecture grid search",
        "notes": "Testing all 96 models of default grid search",
        "run_config": {
            "epochs": 20,
            "save_model": False,
            "class_weights": {0: 1, 1: 5},
            "loss": "binary_crossentropy",
            "metrics": ["accuracy", "precision", "recall"],
        },
        "loader_config": {
            "loader_type": "ts_loader",
            "load_kwargs": {"output_type": "siamese"},
        },
        "builder_config": {"builder_type": "siamese_builder"},
        "grid_config": default_grid_search,
    },
)
job.run()
