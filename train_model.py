import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load preprocessed data
data = pd.read_csv("data/processed_data.csv")
X = data.drop('income', axis=1)
y = data['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "models/final_model.pkl")
print("Model trained and saved as models/final_model.pkl")
