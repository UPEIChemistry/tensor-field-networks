from tfn.tools.jobs import Regression

job = Regression(
    exp_config={
        "name": "QM9 E PREDICTION USING DEFAULT ENERGY MODEL",
        "notes": "Using all defaults",
        "builder_config": {"builder_type": "energy_builder"},
    }
)
job.run()
