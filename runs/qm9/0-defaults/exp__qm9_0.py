from tfn.tools.jobs import SingleModel

job = SingleModel(exp_config={
    'name': 'QM9 E PREDICTION USING DEFAULT ENERGY MODEL',
    'notes': 'Using all defaults'
})
job.run()
