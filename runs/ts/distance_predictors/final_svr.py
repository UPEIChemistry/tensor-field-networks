import pickle
from sklearn.svm import SVR

from tfn.tools.loaders import TSLoader


loader = TSLoader('/home/riley/dev/python/data/ts.hdf5', splitting='90:10')
x, y = loader.load_distance_data()
y0 = y[:, 0]
y1 = y[:, 1]
model0 = SVR(C=1, kernel='rbf', epsilon=0.001, gamma='auto')
model1 = SVR(C=1, kernel='rbf', epsilon=0.0001, gamma='auto')
model0.fit(x, y0)
model1.fit(x, y1)
with open('model0.pickle', 'wb') as file:
    pickle.dump(model0, file)
with open('model1.pickle', 'wb') as file:
    pickle.dump(model1, file)
