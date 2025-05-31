import joblib
import json
import pandas as pd
import numpy as np
import os
from pprint import pprint

# Load model and necessary files
model_path = os.path.join('..', 'models', 'xgboost_model.joblib')
label_encoders_path = os.path.join('..', 'models', 'label_encoders.joblib')
feature_names_path = os.path.join('..', 'models', 'feature_names.json')
categorical_columns_path = os.path.join('..', 'models', 'categorical_columns.json')

model = joblib.load(model_path)
label_encoders = joblib.load(label_encoders_path)

with open(feature_names_path, 'r') as f:
    all_features = json.load(f)
    
with open(categorical_columns_path, 'r') as f:
    categorical_columns = json.load(f)

# Get model features
model_features = model.get_booster().feature_names
print(f"Model requires {len(model_features)} features")

# Print sample values from each categorical encoder
print("\nSample values from categorical encoders:")
for col in categorical_columns:
    if col in label_encoders:
        values = label_encoders[col].classes_
        print(f"{col}: {values[:3]} (total: {len(values)})")

# Create multiple test configurations
test_configurations = [
    {
        'name': 'Budget Office Laptop',
        'brand': 'ACER',
        'ekran kartı': 'Intel UHD Graphics',
        'işlemci modeli': '1005G1',
        'İşletim_Sistemi': 'Windows',
        'ram': 8,
        'ssd': 256,
        'ekran kartı hafızası': 0,
        'temel işlemci hızı': 1.2,
        'maksimum işlemci hızı': 3.4,
        'expected_min': 8000,
        'expected_max': 15000
    },
    {
        'name': 'Gaming Laptop - RTX 3050',
        'brand': 'ASUS',
        'ekran kartı': 'NVIDIA GeForce RTX 3050',
        'işlemci modeli': '12700H',
        'İşletim_Sistemi': 'Windows',
        'ram': 16,
        'ssd': 512,
        'ekran kartı hafızası': 4,
        'temel işlemci hızı': 2.3,
        'maksimum işlemci hızı': 4.7,
        'expected_min': 18000,
        'expected_max': 35000
    },
    {
        'name': 'High-end Gaming Laptop',
        'brand': 'MONSTER',
        'ekran kartı': 'NVIDIA GeForce RTX 4060',
        'işlemci modeli': '13700H',
        'İşletim_Sistemi': 'Windows',
        'ram': 32,
        'ssd': 1024,
        'ekran kartı hafızası': 8,
        'temel işlemci hızı': 2.4,
        'maksimum işlemci hızı': 5.0,
        'expected_min': 35000,
        'expected_max': 60000
    },
    {
        'name': 'MacBook Air M2',
        'brand': 'Apple',
        'ekran kartı': 'Apple M2',
        'işlemci modeli': 'M2',
        'İşletim_Sistemi': 'Mac Os',
        'ram': 16,
        'ssd': 512,
        'ekran kartı hafızası': 0,
        'temel işlemci hızı': 0,
        'maksimum işlemci hızı': 0,
        'expected_min': 25000,
        'expected_max': 45000
    }
]

print(f"\nTesting {len(test_configurations)} laptop configurations...")
print("="*70)

# Convert input to DataFrame
print("\nProcessing input data...")

# Create a clean dataframe with all model features initialized to 0
input_df = pd.DataFrame({feature: [0] for feature in model_features})

# Define feature mapping for lowercase/special character variants
feature_mapping = {
    'brand': 'brand',
    'ekran kartı': 'Ekran_Kartı',
    'işlemci modeli': 'İşlemci_Modeli',
    'İşletim_Sistemi': 'İşletim_Sistemi',
}

# Process categorical features from the input
for input_col, model_col in feature_mapping.items():
    if input_col in sample_input:
        input_value = sample_input[input_col]
        print(f"\nProcessing {input_col} = '{input_value}':")
        
        if model_col in label_encoders:
            encoder = label_encoders[model_col]
            
            # Check if the value exists in encoder classes
            if input_value in encoder.classes_:
                encoded_value = encoder.transform([input_value])[0]
                input_df[model_col] = encoded_value
                print(f"  ✓ Encoded '{input_value}' to {encoded_value}")
            else:
                print(f"  ✗ Value '{input_value}' not in encoder classes")
                print(f"  Available values (sample): {encoder.classes_[:5]}")
                # Keep as zero (default)
        else:
            print(f"  ✗ No encoder found for {model_col}")

# Process numeric features
numeric_features = ['ram', 'ssd', 'ekran kartı hafızası', 'temel işlemci hızı', 'maksimum işlemci hızı']
numeric_mapping = {
    'ram': 'Ram',
    'ssd': 'SSD',
    'ekran kartı hafızası': 'Ekran_Kartı_Hafızası',
    'temel işlemci hızı': 'Temel_İşlemci_Hızı',
    'maksimum işlemci hızı': 'Maksimum_İşlemci_Hızı'
}

print("\nProcessing numeric features:")
for input_col, model_col in numeric_mapping.items():
    if input_col in sample_input:
        value = sample_input[input_col]
        if model_col in model_features:
            input_df[model_col] = value
            print(f"  ✓ Set {model_col} = {value}")
        else:
            print(f"  ✗ {model_col} not in model features")

# Preview the processed dataframe
print("\nProcessed feature dataframe (first few columns):")
print(input_df.iloc[:, :5])

# Make prediction
try:
    # Ensure all numeric columns have numeric dtype
    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    
    prediction = model.predict(input_df)[0]
    print(f"\n✅ Predicted laptop price: {prediction:,.2f} TL")
except Exception as e:
    print(f"\n❌ Error making prediction: {e}")
