#!/usr/bin/env python3
"""
Comprehensive laptop price prediction model accuracy test
"""
import joblib
import json
import pandas as pd
import numpy as np
import os
from pprint import pprint

def load_model_components():
    """Load model and related components"""
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
    
    return model, label_encoders, all_features, categorical_columns

def preprocess_input(sample_input, model, label_encoders, categorical_columns):
    """Preprocess input for model prediction"""
    # Convert to DataFrame
    input_df = pd.DataFrame([sample_input])
    
    # Feature mapping - UI field names to model feature names
    feature_mapping = {
        'brand': 'brand',
        'ekran kartı': 'Ekran_Kartı',
        'işlemci modeli': 'İşlemci_Modeli',
        'ekran kartı hafızası': 'Ekran_Kartı_Hafızası',
        'temel işlemci hızı': 'Temel_İşlemci_Hızı',
        'maksimum işlemci hızı': 'Maksimum_İşlemci_Hızı',
        'İşletim_Sistemi': 'İşletim_Sistemi',
        'ram': 'Ram',
        'ssd': 'SSD'
    }
    
    # Apply feature mapping
    for ui_field, model_field in feature_mapping.items():
        if ui_field in input_df.columns:
            input_df[model_field] = input_df[ui_field]
    
    # Process categorical variables
    for col in categorical_columns:
        if col in input_df.columns and input_df[col].iloc[0] is not None:
            value = input_df[col].iloc[0]
            print(f"\nProcessing {col} = '{value}':")
            
            if col in label_encoders:
                if value in label_encoders[col].classes_:
                    input_df[col] = label_encoders[col].transform([value])
                    print(f"  ✓ Encoded '{value}' to {input_df[col].iloc[0]}")
                else:
                    print(f"  ✗ Value '{value}' not in encoder classes")
                    available_values = label_encoders[col].classes_[:5]
                    print(f"  Available values (sample): {available_values}")
                    # Use most common value
                    values, counts = np.unique(label_encoders[col].classes_, return_counts=True)
                    most_common = values[counts.argmax()]
                    input_df[col] = label_encoders[col].transform([most_common])
                    print(f"  ⚠️  Using most common value: '{most_common}'")
    
    # Process numeric features
    numeric_features = ['Ram', 'SSD', 'Ekran_Kartı_Hafızası', 'Temel_İşlemci_Hızı', 'Maksimum_İşlemci_Hızı']
    print(f"\nProcessing numeric features:")
    for col in numeric_features:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            print(f"  ✓ Set {col} = {input_df[col].iloc[0]}")
    
    # Fill missing values
    input_df = input_df.fillna(0)
    
    # Add missing features expected by model
    model_features = model.get_booster().feature_names
    missing_features = set(model_features) - set(input_df.columns)
    for feature in missing_features:
        input_df[feature] = 0
    
    # Reorder columns to match model expectations
    input_df = input_df[model_features]
    
    return input_df

def test_laptop_configuration(config, model, label_encoders, categorical_columns):
    """Test a single laptop configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {config['name']}")
    print(f"{'='*60}")
    
    # Extract config without metadata
    test_input = {k: v for k, v in config.items() if k not in ['name', 'expected_min', 'expected_max']}
    
    print("Configuration:")
    for key, value in test_input.items():
        print(f"  {key}: {value}")
    
    # Preprocess input
    processed_input = preprocess_input(test_input, model, label_encoders, categorical_columns)
    
    print(f"\nProcessed feature dataframe (first few columns):")
    print(processed_input.iloc[:, :5])
    
    # Make prediction
    prediction = model.predict(processed_input)[0]
    
    # Check accuracy
    expected_min = config.get('expected_min', 0)
    expected_max = config.get('expected_max', float('inf'))
    is_accurate = expected_min <= prediction <= expected_max
    
    print(f"\n{'='*40}")
    print(f"💰 Predicted Price: {prediction:,.2f} TL")
    print(f"📊 Expected Range: {expected_min:,} - {expected_max:,} TL")
    
    if is_accurate:
        print(f"✅ ACCURATE - Prediction within expected range")
        accuracy_status = "ACCURATE"
    else:
        if prediction < expected_min:
            error_pct = ((expected_min - prediction) / expected_min) * 100
            print(f"❌ TOO LOW - {error_pct:.1f}% below minimum")
            accuracy_status = "TOO_LOW"
        else:
            error_pct = ((prediction - expected_max) / expected_max) * 100
            print(f"❌ TOO HIGH - {error_pct:.1f}% above maximum")
            accuracy_status = "TOO_HIGH"
    
    return {
        'name': config['name'],
        'prediction': prediction,
        'expected_min': expected_min,
        'expected_max': expected_max,
        'is_accurate': is_accurate,
        'accuracy_status': accuracy_status
    }

def main():
    print("🔍 Loading model components...")
    model, label_encoders, all_features, categorical_columns = load_model_components()
    
    model_features = model.get_booster().feature_names
    print(f"✅ Model loaded successfully - expects {len(model_features)} features")
    
    # Test configurations
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
    
    # Test all configurations
    results = []
    for config in test_configurations:
        try:
            result = test_laptop_configuration(config, model, label_encoders, categorical_columns)
            results.append(result)
        except Exception as e:
            print(f"❌ Error testing {config['name']}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📈 MODEL ACCURACY SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    accurate_predictions = sum(1 for r in results if r['is_accurate'])
    accuracy_rate = (accurate_predictions / total_tests * 100) if total_tests > 0 else 0
    
    print(f"✅ Accurate Predictions: {accurate_predictions}/{total_tests}")
    print(f"📊 Overall Accuracy: {accuracy_rate:.1f}%")
    
    if results:
        predictions = [r['prediction'] for r in results]
        print(f"💰 Average Predicted Price: {np.mean(predictions):,.2f} TL")
        print(f"📊 Price Range: {np.min(predictions):,.2f} - {np.max(predictions):,.2f} TL")
        
        # Show inaccurate predictions
        inaccurate = [r for r in results if not r['is_accurate']]
        if inaccurate:
            print(f"\n⚠️  INACCURATE PREDICTIONS:")
            for result in inaccurate:
                print(f"  • {result['name']}: {result['prediction']:,.2f} TL ({result['accuracy_status']})")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if accuracy_rate >= 80:
        print("  ✅ Model performance is excellent!")
    elif accuracy_rate >= 60:
        print("  ⚠️  Model performance is acceptable but could be improved")
        print("  💡 Consider feature engineering or more training data")
    else:
        print("  ❌ Model performance needs significant improvement")
        print("  💡 Recommend model retraining with better data quality")

if __name__ == "__main__":
    main()
