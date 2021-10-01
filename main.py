
import numpy as numpy
import pandas as pandas

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

dataset = pandas.read_csv('train.csv')
features = pandas.read_csv('features.csv')

target = numpy.where((dataset.resp_1 > 0) & (dataset.resp_2 > 0) & (dataset.resp_3 > 0) &
                     (dataset.resp_4 > 0) & (dataset.resp > 0), 1, 0)
train = dataset[['date', 'weight', 'resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp']]
train = StandardScaler().fit_transform(train)

# n_neighbors = [{"n_neighbors": [4, 6, 8, 10, 12]}]  8
knn = KNeighborsRegressor(n_neighbors=8)
knn.fit(train, target)
pred = knn.predict(train)
print(mean_squared_error(target, pred))

# needed to tune n_neighbors parameter
# meta = GridSearchCV(knn, n_neighbors, cv=5, scoring="neg_root_mean_squared_error", refit=True)
# meta.fit(train, target)
# print(meta.best_params_)

# meta = GridSearchCV(knn, n_neighbors, cv=5, scoring="neg_root_mean_squared_error", refit=True)
# knn_score = 0 - cross_val_score(meta, train, target, cv=10, scoring="neg_root_mean_squared_error")
# print(knn_score.mean())

# knn = KNeighborsClassifier()
# knn.fit(train, target)
# pred = knn.predict(train)
# print(accuracy_score(target, pred))

# meta = GridSearchCV(knn, cv=5, scoring="roc_auc", refit=True)
# knn_score = 0 - cross_val_score(meta, train, target, cv=10, scoring="roc_auc")
# print(knn_score.mean())
