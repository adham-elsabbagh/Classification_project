# Support Vector Machine (SVM)

# Importing the libraries
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

# Importing the dataset

missing_values=['?']
potential = pd.read_csv("potential-clients.csv", index_col="ID",na_values = missing_values)
current = pd.read_csv("current-clients.csv", index_col="ID",na_values = missing_values)
# fill the missing values with the mod
potential['workclass'].fillna(mode(potential['workclass']).mode[0],inplace=True)
potential['native-country'].fillna(mode(potential['native-country']).mode[0],inplace=True)
potential['occupation'].fillna(mode(potential['occupation']).mode[0],inplace=True)
current['occupation'].fillna(mode(current['occupation']).mode[0],inplace=True)
current['workclass'].fillna(mode(current['workclass']).mode[0],inplace=True)
current['native-country'].fillna(mode(current['native-country']).mode[0],inplace=True)
print (current.isnull().sum())

# drop missing values
current.dropna(inplace=True)
potential.dropna(inplace=True)

labelencoder = LabelEncoder()
cols = ['workclass' , 'education' , 'marital-status' , 'occupation' , 'sex' , 'native-country' , 'race' , 'relationship','class']
for col_name in cols:
    current[col_name] = labelencoder.fit_transform(current[col_name])
    # potential[col_name] = labelencoder.fit_transform(potential[col_name])

print(current.head())
y = current.pop('class').values
X = current.values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.20, random_state = 50,shuffle=True)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting XGBClassifier to the Training set
classifier =  XGBClassifier(n_estimators=60, n_jobs=-1, random_state=50)
classifier.fit(X_train, y_train)
accuracy=classifier.score(X_test,y_test)
print('accuracy: ',accuracy)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cr=classification_report(y_test, y_pred)
print(cm)
print(cr)



# Visualising the Training set results
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(11)]).T
pred = classifier.predict(Xpred).reshape(X1.shape)   # is a matrix of 0's and 1's !
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.contourf(X1, X2, pred,
#              alpha=0.75, cmap=ListedColormap(('red', 'green')), )
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('XGBClassifier (Training set)')
plt.xlabel('Features')
plt.ylabel('Estimated Salary')
plt.legend()
plt.savefig('Training set')
plt.show()
# Visualising the Test set results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.contourf(X1, X2, pred, alpha=0.75, cmap=ListedColormap(('red', 'green')), )
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('yellow', 'black'))(i), label = j)
plt.title('XGBClassifier (Test set)')
plt.xlabel('Features')
plt.ylabel('Estimated Salary')
plt.legend()
plt.savefig('Test set')
plt.show()
