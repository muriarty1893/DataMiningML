import joblib
import json
import pandas as pd
import numpy as np
import os

# Load model and necessary files
model_path = os.path.join('..', 'models', 'xgboost_model.joblib')
label_encoders_path = os.path.join('..', 'models', 'label_encoders.joblib')

model = joblib.load(model_path)
label_encoders = joblib.load(label_encoders_path)

def predict_price(laptop_config):
    """Predict the price of a laptop based on its configuration."""
    # Create a dataframe with all features set to 0
    input_df = pd.DataFrame({feature: [0] for feature in model.get_booster().feature_names})
    
    # Feature mapping
    feature_mapping = {
        'brand': 'brand',
        'ekran kartı': 'Ekran_Kartı',
        'işlemci modeli': 'İşlemci_Modeli',
        'İşletim_Sistemi': 'İşletim_Sistemi',
    }
    
    # Process categorical features
    for input_col, model_col in feature_mapping.items():
        if input_col in laptop_config and model_col in label_encoders:
            input_value = laptop_config[input_col]
            if input_value in label_encoders[model_col].classes_:
                encoded_value = label_encoders[model_col].transform([input_value])[0]
                input_df[model_col] = encoded_value
    
    # Process numeric features
    numeric_mapping = {
        'ram': 'Ram',
        'ssd': 'SSD',
        'ekran kartı hafızası': 'Ekran_Kartı_Hafızası',
        'temel işlemci hızı': 'Temel_İşlemci_Hızı',
        'maksimum işlemci hızı': 'Maksimum_İşlemci_Hızı'
    }
    
    for input_col, model_col in numeric_mapping.items():
        if input_col in laptop_config and model_col in input_df.columns:
            input_df[model_col] = laptop_config[input_col]
    
    # Ensure all columns are numeric
    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    return prediction

# Feature mapping to use in testing
feature_mapping = {
    'brand': 'brand',
    'ekran kartı': 'Ekran_Kartı',
    'işlemci modeli': 'İşlemci_Modeli',
    'İşletim_Sistemi': 'İşletim_Sistemi',
}

# Test configs
test_configs = [
    {
        'name': 'Budget Gaming Laptop',
        'config': {
            'brand': 'MONSTER', 
            'işlemci modeli': '12500H',  # Mid-tier CPU
            'ekran kartı': 'NVIDIA GeForce RTX 3050',  # Entry gaming GPU
            'İşletim_Sistemi': 'Windows',
            'ram': 8,
            'ssd': 256,
            'ekran kartı hafızası': 4,
            'temel işlemci hızı': 2.5,
            'maksimum işlemci hızı': 4.2,
        }
    },
    {
        'name': 'Mid-range Gaming Laptop',
        'config': {
            'brand': 'MSI', 
            'işlemci modeli': '12700H',  # High-tier CPU
            'ekran kartı': 'NVIDIA GeForce RTX 3060',  # Better GPU
            'İşletim_Sistemi': 'Windows',
            'ram': 16,
            'ssd': 512,
            'ekran kartı hafızası': 6,
            'temel işlemci hızı': 2.3,
            'maksimum işlemci hızı': 4.7,
        }
    },
    {
        'name': 'High-end Gaming Laptop',
        'config': {
            'brand': 'ASUS', 
            'işlemci modeli': '13700H',  # Top-tier CPU
            'ekran kartı': 'NVIDIA GeForce RTX 4060',  # Top-tier GPU
            'İşletim_Sistemi': 'Windows',
            'ram': 32,
            'ssd': 1024,
            'ekran kartı hafızası': 8,
            'temel işlemci hızı': 2.6,
            'maksimum işlemci hızı': 5.0,
        }
    },
    {
        'name': 'MacBook',
        'config': {
            'brand': 'Apple', 
            'işlemci modeli': 'M2',  # Apple CPU
            'ekran kartı': 'Apple M2',  # Apple GPU
            'İşletim_Sistemi': 'Mac Os',
            'ram': 16,
            'ssd': 512,
            'ekran kartı hafızası': 0,  # Integrated
            'temel işlemci hızı': 0,  # Different architecture
            'maksimum işlemci hızı': 0,  # Different architecture
        }
    }
]

print("\n===== Testing Multiple Laptop Configurations =====\n")

for test in test_configs:
    print(f"Testing: {test['name']}")        # Check if the features exist in our label encoders (for categorical features)
    print(f"  Configuration: {test['config']}")
    issues = []
    for key, value in test['config'].items():
        if key in ['brand', 'ekran kartı', 'işlemci modeli', 'İşletim_Sistemi']:
            mapped_key = feature_mapping.get(key, key)
            print(f"  Checking {key}='{value}' (maps to {mapped_key})")
            if mapped_key in label_encoders:
                if value not in label_encoders[mapped_key].classes_:
                    issues.append(f"Value '{value}' not found in {mapped_key} encoder")
    
    if issues:
        print(f"  Warning: {', '.join(issues)}")
        print("  Finding closest matches...")
        
        # Try to find closest matches for categorical features with issues
        for key, value in test['config'].items():
            if key in ['brand', 'ekran kartı', 'işlemci modeli', 'İşletim_Sistemi']:
                mapped_key = feature_mapping.get(key, key)
                if mapped_key in label_encoders:
                    if value not in label_encoders[mapped_key].classes_:
                        # Just show some available options
                        print(f"  For {key}, instead of '{value}' try one of these:")
                        print(f"    {list(label_encoders[mapped_key].classes_)[:5]}")
    
    # Predict price anyway
    try:
        predicted_price = predict_price(test['config'])
        print(f"  Predicted price: {predicted_price:,.2f} TL")
    except Exception as e:
        print(f"  Error predicting price: {e}")
    
    print()

print("===== End of Tests =====")
