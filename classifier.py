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

# print(current.groupby('education').size())
# print (current.isnull().sum())

potential['workclass'].fillna('Private',inplace=True)
potential['native-country'].fillna('United-States',inplace=True)
current['workclass'].fillna('Private',inplace=True)
current['native-country'].fillna('United-States',inplace=True)

current.dropna(inplace=True)
potential.dropna(inplace=True)
new_pot=potential
new_pot.to_csv('complete_poten')



# Feature Engineering

labelencoder = LabelEncoder()
# current file preprocessing
current['workclass'] = labelencoder.fit_transform(current['workclass'])
current['education'] = labelencoder.fit_transform(current['education'])
current['marital-status'] = labelencoder.fit_transform(current['marital-status'])
current['occupation'] = labelencoder.fit_transform(current['occupation'])
current['sex'] = labelencoder.fit_transform(current['sex'])
current['native-country'] = labelencoder.fit_transform(current['native-country'])
current['race'] = labelencoder.fit_transform(current['race'])
current['relationship'] = labelencoder.fit_transform(current['relationship'])

# current=current.drop(columns = ['workclass', 'education', 'marital-status', 'occupation', 'sex',
#                                 'native-country','race','relationship'],axis = 1)

# potential file preprocessing
potential['workclass'] = labelencoder.fit_transform(potential['workclass'])
potential['education'] = labelencoder.fit_transform(potential['education'])
potential['marital-status'] = labelencoder.fit_transform(potential['marital-status'])
potential['occupation'] = labelencoder.fit_transform(potential['occupation'])
potential['sex'] = labelencoder.fit_transform(potential['sex'])
potential['native-country'] = labelencoder.fit_transform(potential['native-country'])
potential['race'] = labelencoder.fit_transform(potential['race'])
potential['relationship'] = labelencoder.fit_transform(potential['relationship'])

# potential=potential.drop(columns=['workclass', 'education', 'marital-status', 'occupation', 'sex',
#                                 'native-country','race','relationship'],axis=1,)
# potential.to_csv('complete_poten')
# print(current.describe())
# print(current.head())
# print(potential.head())

# Model Building

x = np.array(current.drop(['class'], 1))# features
y = np.array(current['class'])# Label
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2,stratify=y)

clf = KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print('KNN acurecy',accuracy*100 ,'%')

pred = clf.predict(potential)
print(pred)
potential['class']=list(pred)
potential.to_csv('newPotential.csv')

new_potential = pd.read_csv("newPotential.csv", index_col="ID")
# print(new_potential.groupby('class').size())


file = open("classes2.txt","w")
file.write('ID'+' '+'Class'+'\n')
for i,c in new_potential.iterrows():
    if c['class']=='>50K':
        high_income=c['class']
        # print(str(i),high_income)
        file.write(str(i)+'  '+high_income+'\n')

    else :
        low_income=c['class']
        # print(str(i),low_income)
        file.write(str(i)+'  '+low_income+'\n')
file.close()

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
# evaluate each model in turn
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


