#!/usr/bin/env python3
"""
Simple test to identify the exact XGBoost data type issue
"""

import pandas as pd
import joblib
import json
import os
import numpy as np

# Load model and encoders
model_path = os.path.join('..', 'models', 'xgboost_model.joblib')
label_encoders_path = os.path.join('..', 'models', 'label_encoders.joblib')
categorical_columns_path = os.path.join('..', 'models', 'categorical_columns.json')

model = joblib.load(model_path)
label_encoders = joblib.load(label_encoders_path)

with open(categorical_columns_path, 'r') as f:
    categorical_columns = json.load(f)

print("Available label encoders:", list(label_encoders.keys()))
print("Categorical columns:", categorical_columns)
print("Model expected features:", model.get_booster().feature_names)

# Test with simple data
test_data = {
    'brand': 'Apple',
    'Ekran_Kartı': 'Apple M1',
    'İşlemci_Modeli': 'Apple M1',
    'İşletim_Sistemi': 'macOS',
    'Ram': 8,
    'SSD': 256,
    'Ekran_Kartı_Hafızası': 0,
    'Temel_İşlemci_Hızı': 2.4,
    'Maksimum_İşlemci_Hızı': 3.2
}

# Convert to DataFrame
input_data = pd.DataFrame([test_data])
print("\nOriginal input data:")
print(input_data.dtypes)

# Encode categorical variables
for column in categorical_columns:
    if column in input_data.columns and input_data[column].iloc[0] is not None:
        value = input_data[column].iloc[0]
        if column in label_encoders:
            try:
                if value in label_encoders[column].classes_:
                    encoded_value = label_encoders[column].transform([value])[0]
                    input_data[column] = encoded_value
                    print(f"Encoded {column}: {value} -> {encoded_value} (type: {type(encoded_value)})")
                else:
                    # Use first available class
                    first_class = label_encoders[column].classes_[0]
                    encoded_value = label_encoders[column].transform([first_class])[0]
                    input_data[column] = encoded_value
                    print(f"Not found, using first class for {column}: {first_class} -> {encoded_value}")
            except Exception as e:
                print(f"Error encoding {column}: {e}")
                input_data[column] = 0

# Convert all categorical columns to numeric
for column in categorical_columns:
    if column in input_data.columns:
        print(f"Before conversion - {column}: {input_data[column].dtype}")
        input_data[column] = pd.to_numeric(input_data[column], errors='coerce')
        print(f"After conversion - {column}: {input_data[column].dtype}")

# Fill missing values
input_data = input_data.fillna(0)

# Convert all to float
for column in input_data.columns:
    input_data[column] = input_data[column].astype(float)

print("\nFinal input data types:")
print(input_data.dtypes)

# Add missing features
model_features = model.get_booster().feature_names
missing_features = set(model_features) - set(input_data.columns)
for feature in missing_features:
    input_data[feature] = 0.0

# Reorder columns
input_data = input_data[model_features]

print(f"\nFinal data shape: {input_data.shape}")
print("First few values:")
print(input_data.iloc[0][:10])

# Try prediction
try:
    prediction = model.predict(input_data)[0]
    print(f"\nPrediction successful: {prediction}")
except Exception as e:
    print(f"\nPrediction failed: {e}")
    print(f"Error type: {type(e)}")
    
    # Check data types in detail
    print("\nDetailed data type analysis:")
    for col in input_data.columns[:10]:  # Check first 10 columns
        print(f"{col}: {input_data[col].dtype}, value: {input_data[col].iloc[0]}")
