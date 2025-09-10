from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def knnClassifier(X_train, y_train, x_test, y_test):
    nbr = KNeighborsClassifier(3)
    nbr.fit(X_train, y_train)
    y_pred = nbr.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
    # print("Knn = ", accuracy)
    
