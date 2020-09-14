from pathlib import Path
from tfn.tools.jobs import Pipeline


job = Pipeline(
    {
        "name": f"{Path(__file__).parent}",
        "notes": "",
        "pipeline_config": {
            "configs": [
                {  # SN2 config
                    "builder_config": {"builder_type": "force_builder"},
                    "loader_config": {"loader_type": "sn2_loader"},
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
