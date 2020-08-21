from pathlib import Path
from tfn.tools.jobs import CrossValidate


job = CrossValidate(
    exp_config={
        "name": f"{Path(__file__).parent}",
        "notes": "",
        "loader_config": {"loader_type": "ts_loader", "splitting": 5},
        "builder_config": {
            "builder_type": "cartesian_builder",
            "prediction_type": "cartesians",
            "output_type": "cartesians",
        },
    }
)
job.run()
