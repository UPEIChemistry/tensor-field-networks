from pathlib import Path
from tfn.tools.jobs import Pipeline


job = Pipeline(
    {
        "name": f"{Path(__file__).parent}",
        "notes": "",
        "pipeline_config": {
            "configs": [
                {  # ISO17 config
                    "builder_config": {"builder_type": "force_builder"},
                    "loader_config": {"loader_type": "iso17_loader"},
                },
                {  # QM9 config
                    "builder_config": {"builder_type": "energy_builder"},
                    "loader_config": {"loader_type": "qm9_loader"},
                },
            ]
        },
    }
)
job.run()
