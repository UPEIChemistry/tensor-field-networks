from pathlib import Path
from tfn.tools.jobs import Pipeline


job = Pipeline(
    {
        "name": f"{Path(__file__).parent}",
        "notes": "",
        "pipeline_config": {
            "configs": [
                {  # QM9 config
                    "builder_config": {"builder_type": "energy_builder"},
                    "loader_config": {"loader_type": "qm9_loader"},
                },
                {  # TS config
                    "builder_config": {
                        "builder_type": "cartesian_builder",
                        "prediction_type": "vectors",
                        "output_types": "cartesians",
                    },
                    "loader_config": {"loader_type": "ts_loader"},
                },
            ]
        },
    }
)
job.run()
