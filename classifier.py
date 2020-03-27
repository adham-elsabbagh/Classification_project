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


potential['workclass'].fillna('Private',inplace=True)
potential['native-country'].fillna('United-States',inplace=True)
current['workclass'].fillna('Private',inplace=True)
current['native-country'].fillna('United-States',inplace=True)

current.dropna(inplace=True)
potential.dropna(inplace=True)
new_pot=potential
new_pot.to_csv('complete_poten')

# print(current.groupby('occupation').size())
# print (current.isnull().sum())

# Feature Engineering

labelencoder = LabelEncoder()
# current file preprocessing
current['workclass_cat'] = labelencoder.fit_transform(current['workclass'])
current['education_cat'] = labelencoder.fit_transform(current['education'])
current['marital-status_cat'] = labelencoder.fit_transform(current['marital-status'])
current['occupation_cat'] = labelencoder.fit_transform(current['occupation'])
current['sex_cat'] = labelencoder.fit_transform(current['sex'])
current['native-country_cat'] = labelencoder.fit_transform(current['native-country'])
current['race_cat'] = labelencoder.fit_transform(current['race'])
current['relationship_cat'] = labelencoder.fit_transform(current['relationship'])

current=current.drop(columns = ['workclass', 'education', 'marital-status', 'occupation', 'sex',
                                'native-country','race','relationship'],axis = 1)

# potential file preprocessing
potential['workclass_cat'] = labelencoder.fit_transform(potential['workclass'])
potential['education_cat'] = labelencoder.fit_transform(potential['education'])
potential['marital-status_cat'] = labelencoder.fit_transform(potential['marital-status'])
potential['occupation_cat'] = labelencoder.fit_transform(potential['occupation'])
potential['sex_cat'] = labelencoder.fit_transform(potential['sex'])
potential['native-country_cat'] = labelencoder.fit_transform(potential['native-country'])
potential['race_cat'] = labelencoder.fit_transform(potential['race'])
potential['relationship_cat'] = labelencoder.fit_transform(potential['relationship'])

potential=potential.drop(columns=['workclass', 'education', 'marital-status', 'occupation', 'sex',
                                'native-country','race','relationship'],axis=1,)
# potential.to_csv('complete_poten')
# print(current.describe())
# print(current.head())
# print(potential.head())

# Model Building

x = np.array(current.drop(['class'], 1))# features
y = np.array(current['class'])# Label
# X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2,stratify=y)
X_test = potential

clf = KNeighborsClassifier()
clf.fit(x,y)

accuracy=clf.score(x,y)
# print('KNN acurecy',accuracy*100 ,'%')

pred = clf.predict(X_test)
# print(pred)
potential['class']=list(pred)
potential.to_csv('newPotential.csv')

new_potential = pd.read_csv("newPotential.csv", index_col="ID")
# print(new_potential.groupby('class').size())
# new_potential['cat_class'] = labelencoder.fit_transform(new_potential['class'])
# print(new_potential['cat_class'])

# for i in new_potential['cat_class']:
#     if i ==1:
#         print(i)
for i,c in new_potential.iterrows():
    if c['class']=='>50K':
        high_income=c['class']

        print(i,high_income)
    else :
        low_income=c['class']
        # print(i,low_income)
# print(confusion_matrix(y_test, pred))
# print(classification_report(y_test, pred))

# example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
# example_measures=example_measures.reshape(len(example_measures),-1)
# prediction = clf.predict(example_measures)
# print(prediction)

# prepare configuration for cross validation test harness
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



# print(potential['workclass_cat'].head(10))
# print(current['workclass_cat'].head(10))
