import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv('cardio_train.csv', sep=';')

# Convert age to years
df['age'] = (df['age'] / 365).astype(int)

# Calculate BMI
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

# Drop id
df = df.drop('id', axis=1)

# Features
numeric_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

X_numeric = df[numeric_features]
X_categorical = df[categorical_features]
y = df['cardio']

# Scale numeric features
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Save scaler params
X_mean = scaler.mean_
X_std = scaler.scale_

np.save('X_mean.npy', X_mean)
np.save('X_std.npy', X_std)

# Combine features
X = np.concatenate([X_categorical.values, X_numeric_scaled], axis=1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

# Evaluate
y_pred = gb.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# Save model
joblib.dump(gb, 'final_model.pkl')