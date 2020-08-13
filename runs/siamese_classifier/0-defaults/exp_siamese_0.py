from tfn.tools.jobs import SingleModel
import os

os.environ["DATADIR"] = "/home/riley/dev/python/data"

job = SingleModel(
    exp_config={
        "name": "SIAMESE CLASSIFIER ON TS DATASET EXPERIMENT 0",
        "notes": "Using all defaults",
        "run_config": {"class_weights": {0: 1, 1: 5}, "loss": "binary_crossentropy"},
        "loader_config": {
            "loader_type": "ts_loader",
            "load_kwargs": {"output_type": "siamese"},
        },
        "builder_config": {"builder_type": "siamese_builder"},
    }
)
job.run()
