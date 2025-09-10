import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from knn import knnClassifier
from Svm import SvmClassification
from RandomForest import RandomForestClassification
from DecTree import DecTreeClassification



dataset = pd.read_csv('diabetes.csv')
# print(dataset.head());
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
# print(X);
# print(Y);
X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=0, test_size=0.2)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

LG=LogisticRegression(random_state=0)
LG.fit(X_train,y_train)

y_pred = LG.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred)
dt_accuracy = DecTreeClassification(X_train, y_train, X_test, y_test)
kn_accuracy = knnClassifier(X_train, y_train, X_test, y_test)
svm_accuracy = SvmClassification(X_train, y_train, X_test, y_test)
rf_accuracy = RandomForestClassification(X_train, y_train, X_test, y_test)

technique = ['Logistic', "Random forest", "knn", "SVM", "Decision Tree"]
performance = [lr_accuracy, rf_accuracy, kn_accuracy, svm_accuracy, dt_accuracy]

print("Logistic= " , lr_accuracy)
print("Random forest = " , rf_accuracy)
print("knn = " , kn_accuracy)
print("SVM = ", svm_accuracy)
print("Decision Tree = ", dt_accuracy)

plt.bar(technique, performance)
plt.title("Model Performance")
plt.xlabel("Techniques")
plt.ylabel("Accuracy")
plt.savefig("file.png")