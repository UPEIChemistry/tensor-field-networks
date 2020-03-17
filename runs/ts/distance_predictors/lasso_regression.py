import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate

from tfn.tools.loaders import TSLoader


loader = TSLoader('/home/riley/dev/python/data/ts.hdf5', splitting='90:10')
x, y = loader.load_distance_data()
model = Lasso()
results = cross_validate(model, x, y, return_estimator=True, return_train_score=True, cv=5)
models = results['estimator']
print('CV average train score: {}'.format(np.mean(results['train_score'])))
print('CV average test score: {}'.format(np.mean(results['test_score'])))
print(
    'Average mean absolute error for models: {}'.format(
        np.mean([mean_absolute_error(y, model.predict(x)) for model in models])
    )
)
