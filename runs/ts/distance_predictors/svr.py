import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

from tfn.tools.loaders import TSLoader


loader = TSLoader('/home/riley/dev/python/data/ts.hdf5', splitting='90:10')
x, y = loader.load_distance_data()
y0 = y[:, 0]
y1 = y[:, 1]
model0 = SVR(C=1, kernel='rbf', epsilon=0.001, gamma='auto')
model1 = SVR(C=1, kernel='rbf', epsilon=0.0001, gamma='auto')
results0 = cross_validate(model0, x, y0, return_estimator=True, return_train_score=True, cv=7)
results1 = cross_validate(model1, x, y1, return_estimator=True, return_train_score=True, cv=7)
models0 = results0['estimator']
models1 = results1['estimator']
print('### Nucleophile to Center ###')
print('CV average train score: {}'.format(np.mean(results0['train_score'])))
print('CV average test score: {}'.format(np.mean(results0['test_score'])))
print('MAE: {}'.format(
        np.mean([mean_absolute_error(y0, model.predict(x)) for model in models0])
))
print('\n### Center to LG ###')
print('CV average train score: {}'.format(np.mean(results1['train_score'])))
print('CV average test score: {}'.format(np.mean(results1['test_score'])))
print('MAE: {}'.format(
    np.mean([mean_absolute_error(y1, model.predict(x)) for model in models1])
))
