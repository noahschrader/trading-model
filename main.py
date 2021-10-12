import numpy as numpy
import pandas as pandas

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
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

knn = KNeighborsClassifier(n_neighbors=1000)
knn_score = cross_val_score(knn, train, target, cv=10, scoring="roc_auc")
print(knn_score.mean())

mn = MultinomialNB()
mn.fit(train, target)
mn_score = cross_val_score(mn, train, target, cv=10, scoring="roc_auc")
print(knn_score.mean())

lr = LogisticRegression(max_iter=7000).fit(train, target)
lr_score = cross_val_score(lr, train, target, cv=10, scoring="roc_auc")
