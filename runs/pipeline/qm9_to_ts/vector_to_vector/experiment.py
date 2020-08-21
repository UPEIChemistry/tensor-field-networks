from pathlib import Path
from tfn.tools.jobs import Pipeline


job = Pipeline(
    {
        "name": f"{Path(__file__).parent}",
        "notes": "",
        "pipeline_config": {
            "configs": [
                {  # QM9 config
                    "builder_config": {
                        "builder_type": "cartesian_builder",
                        "prediction_type": "vectors",
                        "output_type": "cartesians",
                    },
                    "loader_config": {
                        "loader_type": "qm9_loader",
                        "load_kwargs": {"modify_structures": True},
                    },
                },
                {  # TS config
                    "builder_config": {
                        "builder_type": "cartesian_builder",
                        "prediction_type": "vectors",
                        "output_type": "cartesians",
                    },
                    "loader_config": {"loader_type": "ts_loader"},
                },
            ]
        },
    }
)
job.run()
