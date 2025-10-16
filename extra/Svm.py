from sklearn import svm
from sklearn.metrics import accuracy_score

def SvmClassification(X_train, y_train, X_test, y_test):
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
    # print("Svm = " , accuracy)