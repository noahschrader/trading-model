import numpy as numpy
import pandas as pandas

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler

dataset = pandas.read_csv('train.csv')

target = numpy.where((dataset.resp_1 > 0) & (dataset.resp_2 > 0) & (dataset.resp_3 > 0) &
                     (dataset.resp_4 > 0) & (dataset.resp > 0), 1, 0)
train = dataset[['date', 'weight', 'resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp']]
train = StandardScaler().fit_transform(train)

# needed to tune n_neighbors parameter for knn
# n_neighbors = [{"n_neighbors": [4, 6, 8, 10, 12]}]  8
# meta = GridSearchCV(knn, n_neighbors, cv=5, scoring="roc_auc", refit=True)
# meta.fit(train, target)
# print(meta.best_params_)

knn = KNeighborsClassifier()
knn_scores = cross_val_score(knn, train, target, cv=10, scoring='roc_auc')
knn_accuracy = knn_scores.mean()
knn_predictions = cross_val_predict(knn, train, target, cv=10)

lr = LogisticRegression()
lr_scores = cross_val_score(lr, train, target, cv=10, scoring='roc_auc')
lr_accuracy = lr_scores.mean()
lr_predictions = cross_val_predict(lr, train, target, cv=10)

nb = GaussianNB()
nb_scores = cross_val_score(nb, train, target, cv=10, scoring='roc_auc')
nb_accuracy = nb_scores.mean()
nb_predictions = cross_val_predict(nb, train, target, cv=10)

print('--------Accuracy Scores--------')
print('K-Nearest Neighbors: ' + str(round(knn_accuracy * 100, 4)) + ' %')
print('Logistic Regression: ' + str(round(lr_accuracy * 100, 4)) + ' %')
print('Naive Bayes: ' + str(round(nb_accuracy * 100, 4)) + ' %')

# knn model performs the best so format its results
actions = pandas.DataFrame(nb_predictions, columns=['Action'])
results = pandas.concat([dataset['ts_id'], actions], axis=1)
results.to_csv(path_or_buf='results.csv')
