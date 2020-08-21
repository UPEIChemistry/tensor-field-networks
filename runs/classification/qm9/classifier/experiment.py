from pathlib import Path
from tfn.tools.jobs import Classification


job = Classification(
    exp_config={
        "name": f"{Path(__file__).parent}",
        "notes": "",
        "run_config": {"loss": "binary_crossentropy"},
        "loader_config": {
            "loader_type": "qm9_loader",
            "load_kwargs": {"modify_structures": True, "classifier_output": True},
        },
        "builder_config": {"builder_type": "classifier_builder"},
    }
)
job.run()
