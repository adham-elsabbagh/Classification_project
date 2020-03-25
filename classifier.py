import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

import seaborn as sns

# cleaning data
missing_values=['?']
potential = pd.read_csv("potential-clients.csv", index_col="ID",na_values = missing_values)
current = pd.read_csv("current-clients.csv", index_col="ID",na_values = missing_values)

# current.drop(['race','relationship','capital-gain','capital-loss'],1,inplace=True)

current.dropna(inplace=True)
# print(current.groupby('workclass').size())


# plt.show()



# Feature Engineering

labelencoder = LabelEncoder()
current['workclass_cat'] = labelencoder.fit_transform(current['workclass'])
current['education_cat'] = labelencoder.fit_transform(current['education'])
current['marital-status_cat'] = labelencoder.fit_transform(current['marital-status'])
current['occupation_cat'] = labelencoder.fit_transform(current['occupation'])
current['sex_cat'] = labelencoder.fit_transform(current['sex'])
current['native-country_cat'] = labelencoder.fit_transform(current['native-country'])
current['race_cat'] = labelencoder.fit_transform(current['race'])
current['relationship_cat'] = labelencoder.fit_transform(current['relationship'])

current=current.drop(columns = ['workclass', 'education', 'marital-status', 'occupation', 'sex',
                                'native-country','race','relationship',])
# current.fillna(current.mean())
# print(current.describe())
# print(current.columns)
# print (current.isnull().sum())
#
# print(current.head(50))
x = np.array(current.drop(['class'], 1))#features
y = np.array(current['class'])#Label
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)


clf = KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print('KNN acurecy',accuracy*100 ,'%')
# prepare configuration for cross validation test harness
seed = 7
# prepare models
# models = []
# models.append(('LR', LogisticRegression(max_iter=1)))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))
# # evaluate each model in turn
# results = []
# names = []
# scoring = 'accuracy'
# for name, model in models:
#     kfold = model_selection.KFold(n_splits=10)
#     cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)
pred = clf.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
# boxplot algorithm comparison

# sns.countplot(current['workclass'],label="workclasses")

# current.hist(bins=30, figsize=(9,9))
# plt.suptitle("Histogram for each numeric input variable")
# current.plot(kind='box', subplots=True,  sharex=False, sharey=False, figsize=(9,9),title='Box Plot for each input variable')
# plt.savefig('data_hist')

# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.savefig('Algorithm Comparison')
# plt.show()
