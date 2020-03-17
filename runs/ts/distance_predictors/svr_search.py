from functools import partial
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV

from tfn.tools.loaders import TSLoader


loader = TSLoader('/home/riley/dev/python/data/ts.hdf5', splitting='90:10')
x, y = loader.load_distance_data()
y0 = y[:, 0]
y1 = y[:, 1]
search_space = {  # 45 models
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100, 1000],
    'epsilon': [0.1, 0.001, 0.0001]
}
partial_model = partial(GridSearchCV, param_grid=search_space, cv=7, scoring=make_scorer(
    mean_absolute_error,
    greater_is_better=False))
model0 = partial_model(SVR(gamma='auto'))
model1 = partial_model(SVR(gamma='auto'))
model0.fit(x, y0)
model1.fit(x, y1)

print('### Nucleophile to Center Distance ###')
print('score: {}'.format(model0.score(x, y0)))
print('MAE: {}\n'.format(mean_absolute_error(y0, model0.predict(x))))
print('### Center to LG Distance ###')
print('score: {}'.format(model0.score(x, y1)))
print('MAE: {}\n'.format(mean_absolute_error(y1, model1.predict(x))))
print('model0 params: {}'.format(model0.best_params_))
print('model1 params: {}'.format(model1.best_params_))
