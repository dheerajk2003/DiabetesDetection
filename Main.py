import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- Deep Learning Model Function ---
def deepLearningClassification(X_train, y_train, X_test, y_test):
    """
    Builds, trains, and evaluates a deep learning model for classification.
    Returns the accuracy, predicted classes, and predicted probabilities.
    """
    model = Sequential([
        Dense(12, input_dim=X_train.shape[1], activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
    
    # Evaluate and get accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Get probabilities and predictions
    y_prob = model.predict(X_test).ravel() # Flatten to 1D array
    y_pred = (y_prob > 0.5).astype(int) # Threshold at 0.5 for class prediction
    
    return accuracy, y_pred, y_prob

# --- Data Loading and Preprocessing ---
# Load the dataset from a URL to avoid file not found errors
url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
dataset = pd.read_csv(url)

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# --- Model Training and Prediction ---

# 1. Logistic Regression
lg = LogisticRegression(random_state=0)
lg.fit(X_train, y_train)
y_pred_lg = lg.predict(X_test)
y_prob_lg = lg.predict_proba(X_test)[:, 1]
lr_accuracy = accuracy_score(y_test, y_pred_lg)

# 2. K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_prob_knn = knn.predict_proba(X_test)[:, 1]
kn_accuracy = accuracy_score(y_test, y_pred_knn)

# 3. Support Vector Machine (SVM)
svm = SVC(kernel='linear', random_state=0, probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
y_prob_svm = svm.predict_proba(X_test)[:, 1]
svm_accuracy = accuracy_score(y_test, y_pred_svm)

# 4. Random Forest
rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# 5. Decision Tree
dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]
dt_accuracy = accuracy_score(y_test, y_pred_dt)

# 6. Deep Learning
dl_accuracy, y_pred_dl, y_prob_dl = deepLearningClassification(X_train, y_train, X_test, y_test)


# --- Plotting ---

# 1. Bar chart for model accuracies
plt.figure(figsize=(10, 6))
techniques = ['Logistic Regression', "Random Forest", "KNN", "SVM", "Decision Tree", "Deep Learning"]
performance = [lr_accuracy, rf_accuracy, kn_accuracy, svm_accuracy, dt_accuracy, dl_accuracy]
s = pd.Series(performance, index=techniques).sort_values(ascending=False)
sns.barplot(x=s.index, y=s.values)
plt.title("Model Performance Comparison")
plt.xlabel("Techniques")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_accuracies_with_dl.png")
plt.close()

# 2. Confusion Matrices
models = {
    'Logistic Regression': y_pred_lg,
    'KNN': y_pred_knn,
    'SVM': y_pred_svm,
    'Random Forest': y_pred_rf,
    'Decision Tree': y_pred_dt,
    'Deep Learning': y_pred_dl
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
fig.suptitle('Confusion Matrices for All Models', fontsize=20)

for i, (model_name, y_pred) in enumerate(models.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
    axes[i].set_title(model_name)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("confusion_matrices_with_dl.png")
plt.close()


# 3. ROC Curves
plt.figure(figsize=(10, 8))
models_prob = {
    'Logistic Regression': y_prob_lg,
    'KNN': y_prob_knn,
    'SVM': y_prob_svm,
    'Random Forest': y_prob_rf,
    'Decision Tree': y_prob_dt,
    'Deep Learning': y_prob_dl
}
for model_name, y_prob in models_prob.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.savefig("roc_curves_with_dl.png")
plt.close()

# 4. Precision-Recall Curves
plt.figure(figsize=(10, 8))
for model_name, y_prob in models_prob.items():
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=f'{model_name}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc="lower left")
plt.savefig("precision_recall_curves_with_dl.png")
plt.close()

print("All 4 comparison plots including the deep learning model have been generated and saved.")
print(f"Accuracies:\n{s}")