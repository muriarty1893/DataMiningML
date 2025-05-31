#!/usr/bin/env python3
"""
Model accuracy and performance testing script
Tests various laptop configurations and analyzes model performance
"""

import joblib
import json
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

def load_model_components():
    """Load model and related components"""
    try:
        model_path = os.path.join('..', 'models', 'xgboost_model.joblib')
        label_encoders_path = os.path.join('..', 'models', 'label_encoders.joblib')
        feature_names_path = os.path.join('..', 'models', 'feature_names.json')
        
        model = joblib.load(model_path)
        label_encoders = joblib.load(label_encoders_path)
        
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
            
        return model, label_encoders, feature_names
    except Exception as e:
        print(f"Error loading model components: {e}")
        return None, None, None

def create_test_configurations():
    """Create realistic test configurations with expected price ranges"""
    configurations = [
        # Budget laptops (8,000 - 15,000 TL)
        {
            'name': 'Budget Office Laptop',
            'config': {
                'brand': 'ACER',
                'işlemci modeli': '1005G1',
                'ekran kartı': 'Intel UHD Graphics',
                'ram': 8,
                'ssd': 256,
                'ekran kartı hafızası': 0,
                'temel işlemci hızı': 1.2,
                'maksimum işlemci hızı': 3.4,
                'İşletim_Sistemi': 'Windows'
            },
            'expected_min': 8000,
            'expected_max': 15000
        },
        # Mid-range laptops (15,000 - 30,000 TL)
        {
            'name': 'Gaming Laptop - Mid Range',
            'config': {
                'brand': 'ASUS',
                'işlemci modeli': '12700H',
                'ekran kartı': 'NVIDIA GeForce RTX 3050',
                'ram': 16,
                'ssd': 512,
                'ekran kartı hafızası': 4,
                'temel işlemci hızı': 2.3,
                'maksimum işlemci hızı': 4.7,
                'İşletim_Sistemi': 'Windows'
            },
            'expected_min': 18000,
            'expected_max': 35000
        },
        # High-end laptops (30,000 - 60,000 TL)
        {
            'name': 'High-end Gaming Laptop',
            'config': {
                'brand': 'MONSTER',
                'işlemci modeli': '13700H',
                'ekran kartı': 'NVIDIA GeForce RTX 4060',
                'ram': 32,
                'ssd': 1024,
                'ekran kartı hafızası': 8,
                'temel işlemci hızı': 2.4,
                'maksimum işlemci hızı': 5.0,
                'İşletim_Sistemi': 'Windows'
            },
            'expected_min': 35000,
            'expected_max': 60000
        },
        # MacBook (Premium - 40,000+ TL)
        {
            'name': 'MacBook Air M2',
            'config': {
                'brand': 'Apple',
                'işlemci modeli': 'M2',
                'ekran kartı': 'Apple M2',
                'ram': 16,
                'ssd': 512,
                'ekran kartı hafızası': 0,
                'temel işlemci hızı': 0,
                'maksimum işlemci hızı': 0,
                'İşletim_Sistemi': 'Mac Os'
            },
            'expected_min': 25000,
            'expected_max': 45000
        },
        # Workstation laptop (50,000+ TL)
        {
            'name': 'Workstation Laptop',
            'config': {
                'brand': 'MSI',
                'işlemci modeli': '13900H',
                'ekran kartı': 'NVIDIA GeForce RTX 4070',
                'ram': 32,
                'ssd': 2048,
                'ekran kartı hafızası': 8,
                'temel işlemci hızı': 2.6,
                'maksimum işlemci hızı': 5.4,
                'İşletim_Sistemi': 'Windows'
            },
            'expected_min': 50000,
            'expected_max': 80000
        }
    ]
    return configurations

def preprocess_input(config, model, label_encoders, feature_names):
    """Preprocess input configuration for model prediction"""
    try:
        # Create DataFrame from config
        input_data = pd.DataFrame([config])
        
        # Feature mapping
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
            if ui_field in input_data.columns:
                input_data[model_field] = input_data[ui_field]
        
        # Encode categorical variables
        categorical_columns = list(label_encoders.keys())
        for column in categorical_columns:
            if column in input_data.columns and input_data[column].iloc[0] is not None:
                value = input_data[column].iloc[0]
                
                if column in label_encoders:
                    if value in label_encoders[column].classes_:
                        input_data[column] = label_encoders[column].transform([value])
                    else:
                        # Use most common value if not found
                        values, counts = np.unique(label_encoders[column].classes_, return_counts=True)
                        most_common = values[counts.argmax()]
                        input_data[column] = label_encoders[column].transform([most_common])
                        print(f"  ⚠️  '{value}' not found in {column}, using '{most_common}'")
        
        # Convert numeric features
        numeric_features = ['Ram', 'SSD', 'Ekran_Kartı_Hafızası', 'Temel_İşlemci_Hızı', 'Maksimum_İşlemci_Hızı']
        for column in numeric_features:
            if column in input_data.columns:
                input_data[column] = pd.to_numeric(input_data[column], errors='coerce')
        
        # Fill missing values
        input_data = input_data.fillna(0)
        
        # Add missing features that model expects
        model_features = model.get_booster().feature_names
        missing_features = set(model_features) - set(input_data.columns)
        for feature in missing_features:
            input_data[feature] = 0
        
        # Reorder features to match model expectations
        input_data = input_data[model_features]
        
        return input_data
        
    except Exception as e:
        print(f"Error preprocessing input: {e}")
        return None

def test_model_accuracy():
    """Test model accuracy with various configurations"""
    print("🔍 Loading model components...")
    model, label_encoders, feature_names = load_model_components()
    
    if model is None:
        print("❌ Failed to load model components")
        return
    
    print("✅ Model components loaded successfully")
    print(f"📊 Model expects {len(model.get_booster().feature_names)} features")
    
    # Get test configurations
    test_configs = create_test_configurations()
    
    print(f"\n🧪 Testing {len(test_configs)} laptop configurations...\n")
    
    results = []
    correct_predictions = 0
    total_predictions = len(test_configs)
    
    for i, test_case in enumerate(test_configs, 1):
        print(f"Test {i}: {test_case['name']}")
        print("-" * 50)
        
        # Preprocess input
        processed_input = preprocess_input(test_case['config'], model, label_encoders, feature_names)
        
        if processed_input is None:
            print("❌ Failed to preprocess input")
            continue
        
        # Make prediction
        try:
            prediction = model.predict(processed_input)[0]
            expected_min = test_case['expected_min']
            expected_max = test_case['expected_max']
            
            # Check if prediction is within expected range
            is_accurate = expected_min <= prediction <= expected_max
            if is_accurate:
                correct_predictions += 1
                status = "✅ ACCURATE"
            else:
                status = "❌ INACCURATE"
            
            # Calculate accuracy percentage
            if prediction < expected_min:
                error_pct = ((expected_min - prediction) / expected_min) * 100
                error_type = "TOO LOW"
            elif prediction > expected_max:
                error_pct = ((prediction - expected_max) / expected_max) * 100
                error_type = "TOO HIGH"
            else:
                error_pct = 0
                error_type = "WITHIN RANGE"
            
            print(f"💰 Predicted Price: {prediction:,.2f} TL")
            print(f"📊 Expected Range: {expected_min:,} - {expected_max:,} TL")
            print(f"🎯 Status: {status}")
            if error_pct > 0:
                print(f"📉 Error: {error_pct:.1f}% {error_type}")
            
            # Store results
            results.append({
                'name': test_case['name'],
                'prediction': prediction,
                'expected_min': expected_min,
                'expected_max': expected_max,
                'is_accurate': is_accurate,
                'error_pct': error_pct,
                'error_type': error_type
            })
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
        
        print("\n")
    
    # Summary
    print("=" * 60)
    print("📈 MODEL ACCURACY SUMMARY")
    print("=" * 60)
    
    accuracy_rate = (correct_predictions / total_predictions) * 100
    print(f"✅ Accurate Predictions: {correct_predictions}/{total_predictions}")
    print(f"📊 Overall Accuracy: {accuracy_rate:.1f}%")
    
    if results:
        predictions = [r['prediction'] for r in results]
        print(f"💰 Average Predicted Price: {np.mean(predictions):,.2f} TL")
        print(f"📊 Price Range: {np.min(predictions):,.2f} - {np.max(predictions):,.2f} TL")
        
        # Error analysis
        inaccurate_results = [r for r in results if not r['is_accurate']]
        if inaccurate_results:
            print(f"\n⚠️  INACCURATE PREDICTIONS:")
            for result in inaccurate_results:
                print(f"  • {result['name']}: {result['error_pct']:.1f}% {result['error_type']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if accuracy_rate >= 80:
        print("  ✅ Model performance is good!")
    elif accuracy_rate >= 60:
        print("  ⚠️  Model performance is acceptable but could be improved")
        print("  💡 Consider retraining with more data or feature engineering")
    else:
        print("  ❌ Model performance needs improvement")
        print("  💡 Recommend model retraining with better data quality")
        print("  💡 Check for data drift or feature encoding issues")

if __name__ == "__main__":
    test_model_accuracy()
