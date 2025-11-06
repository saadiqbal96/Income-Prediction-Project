import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Load model and data
clf = joblib.load("models/final_model.pkl")
data = pd.read_csv("data/processed_data.csv")
X = data.drop('income', axis=1)
y = data['income']

# Split test set (optional: you can reuse train_test_split from preprocessing)
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
