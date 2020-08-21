from pathlib import Path
from tfn.tools.jobs import Regression

job = Regression(
    exp_config={
        "name": f"{Path(__file__).parent}",
        "notes": "",
        "loader_config": {"loader_type": "qm9_loader"},
        "builder_config": {"builder_type": "energy_builder"},
    }
)
job.run()
