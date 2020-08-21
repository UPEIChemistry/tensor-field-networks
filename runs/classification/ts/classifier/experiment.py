from pathlib import Path
from tfn.tools.jobs import Classification


job = Classification(
    exp_config={
        "name": f"{Path(__file__).parent}",
        "notes": "",
        "run_config": {"loss": "binary_crossentropy"},
        "loader_config": {
            "loader_type": "ts_loader",
            "load_kwargs": {
                "output_type": "classifier",
                "blacklist": ["reactant", "reactant_complex", "product_complex"],
            },
        },
        "builder_config": {"builder_type": "classifier_builder"},
    }
)
job.run()
