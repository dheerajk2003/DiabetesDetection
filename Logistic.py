import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



dataset = pd.read_csv('diabetes.csv');
# print(dataset.head());
X = dataset.iloc[:,:-1].values;
Y = dataset.iloc[:,-1].values;
# print(X);
# print(Y);
X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=0, test_size=0.2);
# sc_X = StandardScaler();
# X_train = sc_X.fit_transform(X_train);
# X_test = sc_X.transform(X_test);

# LG=LogisticRegression(random_state=0);
# LG.fit(X_train,y_train);

# y_pred = LG.predict(X_test);
# print(y_pred);
# print(y_test);