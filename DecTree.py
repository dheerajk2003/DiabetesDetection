from sklearn import tree
from sklearn.metrics import accuracy_score

def DecTreeClassification(X_train, y_train, x_test, y_test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
    # print("Decision Tree = ",accuracy)