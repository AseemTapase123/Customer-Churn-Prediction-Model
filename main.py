import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# 1. SETUP FOLDERS
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 2. LOAD REAL DATA
# Ensure your file is named exactly this or update the path below
try:
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: CSV file not found in /data folder.")
    exit()

# 3. DATA CLEANING (Critical Industry Step)
# Drop CustomerID - it's just a label, not a feature
df.drop('customerID', axis=1, inplace=True)

# TotalCharges has empty strings that cause errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Fill those 11 missing values with the median
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# 4. ENCODING CATEGORICAL DATA
# Convert 'Yes/No' and other text to numbers
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# 5. SPLIT DATA
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. MODEL TRAINING
print("⏳ Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 7. EVALUATION
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy:.2%}")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 8. VISUALIZATION: Feature Importance
plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Drivers of Customer Churn")
plt.tight_layout()
plt.savefig('outputs/feature_importance.png')
print("\n✅ Visualization saved to outputs/feature_importance.png")

# 9. SAVE MODEL
joblib.dump(model, 'models/churn_model_v1.pkl')
print("✅ Model saved to models/churn_model_v1.pkl")