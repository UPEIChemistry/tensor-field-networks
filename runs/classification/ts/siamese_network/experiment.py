from pathlib import Path
from tfn.tools.jobs import Classification


job = Classification(
    exp_config={
        "name": f"{Path(__file__).parent}",
        "notes": "",
        "run_config": {"class_weights": {0: 1, 1: 5}, "loss": "binary_crossentropy"},
        "loader_config": {
            "loader_type": "ts_loader",
            "load_kwargs": {"output_type": "siamese"},
        },
        "builder_config": {"builder_type": "siamese_builder"},
    }
)
job.run()
