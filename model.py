
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv('/Users/cuixinlei/Desktop/publish/DVSF3.csv')
print(data)
data2=data.drop(data.columns[[0, 1, 2, 3,9]], axis=1)
print(data2,data2.dtypes)
sns.heatmap(data2.corr(), annot=True, cmap ='cool')
plt.show()

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score
from sklearn.naive_bayes import CategoricalNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import kds
from flask import Flask
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# y_train=data2['intention']
X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(data2.drop(['intention'], axis=1),data2['intention'],test_size=0.2, random_state=0)

from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
MLP.fit(X_train, y_train)
pred7=MLP.predict(X_test)
cm7 = confusion_matrix(y_test, pred7)
print('Confusion matrix7:\n', cm7)
accuracy7 = accuracy_score(y_test, pred7)
print('Accuracy7:', accuracy7)
recall7 = recall_score(y_test, pred7,average=None)
print('Recall7:', recall7)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred8=knn.predict(X_test)
pred8_proba=knn.predict_proba(X_test)
print(pred8)
cm8 = confusion_matrix(y_test, pred8)
print('Confusion matrix8:\n', cm8)
accuracy8 = accuracy_score(y_test, pred8)
print('Accuracy8:', accuracy8)
recall8 = recall_score(y_test, pred8,average=None)
print('Recall8:', recall8)


from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
GB.score(X_test, y_test)
pred5=GB.predict(X_test)
cm5 = confusion_matrix(y_test, pred5)
print('Confusion matrix5:\n', cm5)
accuracy5 = accuracy_score(y_test, pred5)
print('Accuracy5:', accuracy5)
recall5 = recall_score(y_test, pred5,average=None)
print('Recall5:', recall5)


from sklearn import tree
DT = tree.DecisionTreeClassifier()
DT= DT.fit(X_train, y_train)
pred4=DT.predict(X_test)
cm4 = confusion_matrix(y_test, pred4)
print('Confusion matrix4:\n', cm4)
accuracy4 = accuracy_score(y_test, pred4)
print('Accuracy4:', accuracy4)
recall4 = recall_score(y_test, pred4,average=None)
print('Recall3:', recall4)

app = Flask(__name__)

# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"
#
# if __name__ == '__main__':
#     app.run()

import pickle
filename = 'savedmodelGB.sav'
pickle.dump(GB,open(filename,'wb'))
load_GB=pickle.load(open(filename,'rb'))
print(load_GB)
GB_test=load_GB.predict(X_test[:1])
print(GB_test)
#print(y_test[:1])

import pickle
filename = 'savedmodelMLP.sav'
pickle.dump(MLP,open(filename,'wb'))
load_MLP=pickle.load(open(filename,'rb'))
print(load_MLP)
MLP_test=load_MLP.predict(X_test[:1])
print(y_test[:1])
print(MLP_test)


print(X_train.columns.values)