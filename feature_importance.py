import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and data
clf = joblib.load("models/final_model.pkl")
data = pd.read_csv("data/processed_data.csv")
X = data.drop('income', axis=1)

# Feature importance
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
top5 = feat_importances.nlargest(5)
print("Top 5 features:\n", top5)

top5.plot(kind='barh')
plt.show()
