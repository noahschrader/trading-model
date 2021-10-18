import numpy as numpy
import pandas as pandas

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

dataset = pandas.read_csv('train.csv')

target = numpy.where((dataset.resp_1 > 0) & (dataset.resp_2 > 0) & (dataset.resp_3 > 0) &
                     (dataset.resp_4 > 0) & (dataset.resp > 0), 1, 0)
train = dataset[['date', 'weight', 'resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp']]
train = StandardScaler().fit_transform(train)

# needed to tune n_neighbors parameter for knn
# n_neighbors = [{"n_neighbors": [4, 6, 8, 10, 12]}]  8
# meta = GridSearchCV(knn, n_neighbors, cv=5, scoring="neg_root_mean_squared_error", refit=True)
# meta.fit(train, target)
# print(meta.best_params_)

knn = KNeighborsClassifier()
knn_score = cross_val_score(knn, train, target, cv=10, scoring="roc_auc")
print(knn_score.mean())

lr = LogisticRegression()
lr.fit(train, target)
lr_score = cross_val_score(lr, train, target, cv=10, scoring="roc_auc")
print(lr_score.mean())

nb = GaussianNB()
nb.fit(train, target)
nb_score = cross_val_score(nb, train, target, cv=10, scoring="roc_auc")
print(nb_score.mean())
