import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder,StandardScaler
from xgboost import XGBClassifier
from pandas.plotting import scatter_matrix
import seaborn as sns
from scipy.stats import mode
from matplotlib.colors import ListedColormap
from datetime import datetime
# Reading data
missing_values=['?']
potential = pd.read_csv("potential-clients.csv", index_col="ID",na_values = missing_values)
current = pd.read_csv("current-clients.csv", index_col="ID",na_values = missing_values)
# print(current.head(20))
# print(potential.groupby('occupation').size())

# fill the missing values with the mod
potential['workclass'].fillna(mode(potential['workclass']).mode[0],inplace=True)
potential['native-country'].fillna(mode(potential['native-country']).mode[0],inplace=True)
potential['occupation'].fillna(mode(potential['occupation']).mode[0],inplace=True)
current['occupation'].fillna(mode(current['occupation']).mode[0],inplace=True)
current['workclass'].fillna(mode(current['workclass']).mode[0],inplace=True)
current['native-country'].fillna(mode(current['native-country']).mode[0],inplace=True)
print (current.isnull().sum())

# drop missing values
# current.dropna(inplace=True)
# potential.dropna(inplace=True)


# Feature Engineering
labelencoder = LabelEncoder()
cols = ['workclass' , 'education' , 'marital-status' , 'occupation' , 'sex' , 'native-country','race' , 'relationship']
for col_name in cols:
    current[col_name] = labelencoder.fit_transform(current[col_name])
    potential[col_name] = labelencoder.fit_transform(potential[col_name])

print(current.describe())
print(potential.describe())

# prepare configuration for cross validation test harness
# prepare models
y = current.pop('class')
x = current
seed=50
scoring = 'accuracy'
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier(n_neighbors=8)))
# models.append(('CART', DecisionTreeClassifier(max_depth=5)))
# models.append(('NB', GaussianNB()))
# models.append(('XGB', XGBClassifier(n_estimators=100, random_state=seed)))
# models.append(('SVM', SVC()))
# models.append(('CNN', MLPClassifier(alpha=1, max_iter=1000)))
# models.append(('AdaBoost', AdaBoostClassifier(n_estimators=100)))
# models.append(('Bagging', BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0)))
# models.append(('RF', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)))
# # evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#     kfold = model_selection.KFold(n_splits=20,random_state=seed,shuffle=True)
#     cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)
# #
# # # boxplot algorithm comparison
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.savefig('Algorithm Comparison')
# plt.show()

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.20,random_state=seed,shuffle=True)
#finding Best K
# k_range = range(1, 10)
# k_scores = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = model_selection.cross_val# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.savefig('Algorithm Comparison')
# plt.show()_score(knn, x, y, cv=5, scoring='accuracy')
#     k_scores.append(scores.mean())
# print(k_scores)
# max_k=max(k_range)
# print(max_k)
# plt.plot( k_range ,k_scores)

# plt.savefig('best K ')
# plt.show()
# params = {
#         'min_child_weight': [1, 5, 10],
#         'gamma': [0.5, 1, 1.5, 2, 5,8,10],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5,7,10],
#         "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
#         }
# xgb = XGBClassifier( n_estimators=60, objective='binary:logistic',
#                     silent=True, nthread=1)
# folds = 20
# param_comb = 5
# kfold = model_selection.KFold(n_splits=20, random_state=seed,shuffle=True)
# results = model_selection.cross_val_score(xgb, x, y, cv=kfold,scoring=scoring,error_score='raise')
# # skf = model_selection.StratifiedKFold(n_splits=folds, shuffle = True, random_state = seed)
#
# random_search = model_selection.RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=kfold, verbose=3, random_state=1001 )
#
#
# # Here we go
# start_time = timer(None) # timing starts from this point for "start_time" variable
# random_search.fit(x,y)
# timer(start_time) # timing ends here for "start_time" variable
#
# print('\n All results:')
# print(random_search.cv_results_)
# print('\n Best estimator:')
# print(random_search.best_estimator_)
# print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
# print(random_search.best_score_ * 2 - 1)
# print('\n Best hyperparameters:')
# print(random_search.best_params_)
# result = pd.DataFrame(random_search.cv_results_)
# result.to_csv('xgb-random-grid-search-results-01.csv', index=False)
clf = XGBClassifier(n_estimators=60, n_jobs=-1, random_state=seed)

kfold = model_selection.KFold(n_splits=20, random_state=seed,shuffle=True)
results = model_selection.cross_val_score(clf, x, y, cv=kfold,scoring=scoring,error_score='raise')
print('Accuracy Score:',results.mean())

clf.fit(X_train,y_train)
pred = clf.predict(potential)

potential['class']=list(pred)
print(pred)
# potential.to_csv('newPotential.csv')



