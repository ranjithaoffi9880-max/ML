import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv(r"C:\Users\WIN10\Desktop\python_ml_journey\project1_churn\Churn.csv")

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# EDA
print(df.shape)
print(df['Churn'].value_counts())
print(df.isnull().sum())
print(df.groupby("Contract")["Churn"].value_counts())

# Prepare features
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

# Encode text columns
le = LabelEncoder()
y = le.fit_transform(y)

text_columns = X.select_dtypes(include='object').columns
for col in text_columns:
    X[col] = le.fit_transform(X[col])

print("X shape:", X.shape)
print("y unique values:", np.unique(y))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)
print("Testing size :", X_test.shape)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model trained! ✅")

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))