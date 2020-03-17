from tfn.tools.loaders import TSLoader
from sklearn.metrics import mean_absolute_error
import numpy as np


loader = TSLoader('/home/riley/dev/python/data/ts.hdf5', splitting='90:10')
x, y = loader.load_distance_data()
y_pred = np.empty_like(y)
y_pred[:] = np.mean(y)
print('Mean Predictor MAE: {}'.format(mean_absolute_error(y, y_pred)))
