import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

iris = pd.read_csv("C:\Practice\Lab Test\LabTest\iris_csv.csv")

X = iris.drop('class',axis=1)
y = iris['class']

# Label encoder
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state = 99)

svm = SVC(C= 10, gamma= 0.1, kernel= 'rbf')
svm.fit(X_train, y_train)
y_pred_default = svm.predict(X_test)
print(classification_report(y_test, y_pred_default))

# Save Model
joblib.dump(svm, "models.pkl")
joblib.dump(encoder, "models1.pkl")

# make predictions
# Read models
classifier_loaded = joblib.load("models.pkl")
encoder_loaded = joblib.load("models1.pkl")

# Prediction set
X_manual_test = [[4.0, 4.0, 4.0, 4.0]]
print("X_manual_test", X_manual_test)

prediction_raw = classifier_loaded.predict(X_manual_test)
print("prediction_raw", prediction_raw)

prediction_real = encoder_loaded.inverse_transform(svm.predict(X_manual_test))
print("Real prediction", prediction_real)