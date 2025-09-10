from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def RandomForestClassification(X_train, y_train, x_test, y_test):
    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
    # print("Random Forest = ",accuracy)