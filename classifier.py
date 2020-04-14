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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix
import seaborn as sns

# cleaning data
missing_values=['?']
potential = pd.read_csv("potential-clients.csv", index_col="ID",na_values = missing_values)
current = pd.read_csv("current-clients.csv", index_col="ID",na_values = missing_values)

# print(current.groupby('education').size())
# print (current.isnull().sum())

# potential['workclass'].fillna('Private',inplace=True)
# potential['native-country'].fillna('United-States',inplace=True)
# current['workclass'].fillna('Private',inplace=True)
# current['native-country'].fillna('United-States',inplace=True)
# current=current.drop(columns = ['capital-gain','capital-loss'],axis = 1)
# potential=potential.drop(columns = ['capital-gain','capital-loss'],axis = 1)

current.dropna(inplace=True)
potential.dropna(inplace=True)

# print(current.columns)
# print(potential.columns)

# Feature Engineering

labelencoder = LabelEncoder()
cols = ['workclass' , 'education' , 'marital-status' , 'occupation' , 'sex' , 'native-country' , 'race' , 'relationship']
for col_name in cols:
    current[col_name] = labelencoder.fit_transform(current[col_name])
    potential[col_name] = labelencoder.fit_transform(potential[col_name])

# print(current['capital-gain'].describe())
# print(potential['capital-loss'].describe())
# print(current.head())
# print(potential.head())

# prepare configuration for cross validation test harness
# prepare models
y = current.pop('class')
x = current
models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier(n_neighbors=8)))
# models.append(('CART', DecisionTreeClassifier(max_depth=5)))
# models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel="linear", C=10)))
# models.append(('SVM', SVC()))
# models.append(('CNN', MLPClassifier(alpha=1, max_iter=1000)))
models.append(('AdaBoost', AdaBoostClassifier(n_estimators=100, random_state=0)))
# models.append(('Bagging', BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0)))
# models.append(('Gaussian Process', GaussianProcessClassifier()))
# models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('RF', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
seed=7
for name, model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed)
    cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
#
# # boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.savefig('Algorithm Comparison')
# plt.show()



X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.20)
#finding Best K
# k_range = range(1, 10)
# k_scores = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = model_selection.cross_val_score(knn, x, y, cv=5, scoring='accuracy')
#     k_scores.append(scores.mean())
# print(k_scores)
# max_k=max(k_range)
# print(max_k)
# plt.plot( k_range ,k_scores)

# plt.savefig('best K ')
# plt.show()

clf = AdaBoostClassifier()
clf.fit(X_train,y_train)
#
accuracy=clf.score(X_test,y_test)
print('AdaBoost acurecy',accuracy)

pred = clf.predict(potential)

# print(confusion_matrix(y_test,pred))
# print(classification_report(y_test,pred))
# print(accuracy_score(y_test,pred))
print(pred)
potential['class']=list(pred)

# potential.to_csv('newPotential.csv')




