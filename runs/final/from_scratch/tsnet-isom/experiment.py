import os
from pathlib import Path
from tfn.tools.jobs import StructurePrediction

os.environ["DATADIR"] = "foobar"
job = StructurePrediction(
    exp_config={
        "name": f"{Path(__file__).parent}",
        "notes": "",
        "seed": 1,
        "run_config": {
            "epochs": 100,
            "test": False,
            "batch_size": 32,
            "fit_verbosity": 1,
        },
        "loader_config": {
            "loader_type": "isom_loader",
            "path": "/home/rjackson/dev/tensor-field-networks/data/isomerization/isomerization_dataset.hd5f",
            "splitting": "75:20:5",
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
