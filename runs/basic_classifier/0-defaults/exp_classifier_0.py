from tfn.tools.jobs import SingleModel

job = SingleModel(
    exp_config={
        "name": "BASIC CLASSIFIER ON TS DATASET EXPERIMENT 0",
        "notes": "Using all defaults",
        "run_config": {"class_weights": {0: 1, 1: 5}, "loss": "binary_crossentropy",},
        "loader_config": {
            "loader_type": "ts_loader",
            "load_kwargs": {"output_type": "classifier"},
        },
        "builder_config": {"builder_type": "classifier_builder"},
    }
)
job.run()
