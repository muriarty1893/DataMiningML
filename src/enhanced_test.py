#!/usr/bin/env python3
# Enhanced prediction testing script
import joblib
import json
import pandas as pd
import numpy as np
import os
import sys
from pprint import pprint

# Add simple coloring for terminals that don't support colorama
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

# Define color constants for compatibility
Fore = Colors
Style = type('Style', (), {'RESET': Colors.RESET})

# Force output to flush immediately
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

# Load model and necessary files
model_path = os.path.join('..', 'models', 'xgboost_model.joblib')
label_encoders_path = os.path.join('..', 'models', 'label_encoders.joblib')
feature_names_path = os.path.join('..', 'models', 'feature_names.json')
categorical_columns_path = os.path.join('..', 'models', 'categorical_columns.json')

try:
    model = joblib.load(model_path)
    label_encoders = joblib.load(label_encoders_path)

    with open(feature_names_path, 'r') as f:
        all_features = json.load(f)
        
    with open(categorical_columns_path, 'r') as f:
        categorical_columns = json.load(f)
        
    # Get model features
    model_features = model.get_booster().feature_names
    print(f"{Fore.GREEN}Model loaded successfully with {len(model_features)} features")
    
except Exception as e:
    print(f"{Fore.RED}Error loading model or data: {e}")
    exit(1)

def predict_price(laptop_config, verbose=True):
    """
    Predict laptop price with detailed debug information
    
    Args:
        laptop_config: Dictionary with laptop specifications
        verbose: Whether to print debug info
    
    Returns:
        dict: Prediction result with debug information
    """
    if verbose:
        print(f"\n{Fore.CYAN}==== Processing Configuration ====")
        pprint(laptop_config)
    
    # Create a clean dataframe with all model features initialized to 0
    input_df = pd.DataFrame({feature: [0] for feature in model_features})
    
    # Define feature mapping for lowercase/special character variants
    feature_mapping = {
        'brand': 'brand',
        'ekran kartı': 'Ekran_Kartı',
        'işlemci modeli': 'İşlemci_Modeli',
        'İşletim_Sistemi': 'İşletim_Sistemi',
        'ram': 'Ram',
        'ssd': 'SSD',
        'ekran kartı hafızası': 'Ekran_Kartı_Hafızası',
        'temel işlemci hızı': 'Temel_İşlemci_Hızı',
        'maksimum işlemci hızı': 'Maksimum_İşlemci_Hızı',
        'Ekran_Boyutu': 'Ekran_Boyutu',
        'Çözünürlük': 'Çözünürlük',
        'Çözünürlük_Standartı': 'Çözünürlük_Standartı',
        'Panel_Tipi': 'Panel_Tipi',
        'Ekran_Yenileme_Hızı': 'Ekran_Yenileme_Hızı',
        'Ekran_Kartı_Tipi': 'Ekran_Kartı_Tipi',
        'İşlemci_Çekirdek_Sayısı': 'İşlemci_Çekirdek_Sayısı',
        'Kullanım_Amacı': 'Kullanım_Amacı',
        'Parmak_İzi_Okuyucu': 'Parmak_İzi_Okuyucu'
    }
    
    # Keep track of debug info
    debug_info = {
        "encoding_issues": [],
        "missing_features": [],
        "used_values": {}
    }
    
    # Process categorical features
    for input_col, model_col in feature_mapping.items():
        if input_col in laptop_config:
            input_value = laptop_config[input_col]
            
            if verbose:
                print(f"\n{Fore.YELLOW}Processing {input_col} = '{input_value}' → {model_col}")
            
            if model_col in label_encoders:
                encoder = label_encoders[model_col]
                
                # Check if the value exists in encoder classes
                if input_value in encoder.classes_:
                    encoded_value = encoder.transform([input_value])[0]
                    input_df[model_col] = encoded_value
                    debug_info["used_values"][model_col] = {
                        "input": input_value,
                        "encoded": encoded_value
                    }
                    if verbose:
                        print(f"  {Fore.GREEN}✓ Encoded '{input_value}' to {encoded_value}")
                else:
                    # Try to find similar value (case insensitive)
                    found_match = False
                    for encoder_value in encoder.classes_:
                        if str(input_value).strip().lower() == str(encoder_value).strip().lower():
                            encoded_value = encoder.transform([encoder_value])[0]
                            input_df[model_col] = encoded_value
                            debug_info["used_values"][model_col] = {
                                "input": input_value,
                                "encoded": encoded_value,
                                "matched_to": encoder_value
                            }
                            found_match = True
                            if verbose:
                                print(f"  {Fore.GREEN}✓ Found case-insensitive match: '{encoder_value}' → {encoded_value}")
                            break
                    
                    if not found_match:
                        debug_info["encoding_issues"].append({
                            "field": model_col,
                            "value": input_value,
                            "available_sample": list(encoder.classes_[:5]) + ['...']
                        })
                        if verbose:
                            print(f"  {Fore.RED}✗ Value '{input_value}' not in encoder classes")
                            print(f"  {Fore.YELLOW}Available values (sample): {encoder.classes_[:5]}")
            else:
                debug_info["missing_features"].append(model_col)
                if verbose:
                    print(f"  {Fore.RED}✗ No encoder found for {model_col}")
    
    # Process numeric features
    numeric_mapping = {
        'ram': 'Ram',
        'ssd': 'SSD',
        'ekran kartı hafızası': 'Ekran_Kartı_Hafızası',
        'temel işlemci hızı': 'Temel_İşlemci_Hızı',
        'maksimum işlemci hızı': 'Maksimum_İşlemci_Hızı',
        'Ekran_Yenileme_Hızı': 'Ekran_Yenileme_Hızı',
        'İşlemci_Çekirdek_Sayısı': 'İşlemci_Çekirdek_Sayısı'
    }
    
    if verbose:
        print(f"\n{Fore.YELLOW}Processing numeric features:")
        
    for input_col, model_col in numeric_mapping.items():
        if input_col in laptop_config:
            value = laptop_config[input_col]
            if model_col in model_features:
                input_df[model_col] = value
                debug_info["used_values"][model_col] = value
                if verbose:
                    print(f"  {Fore.GREEN}✓ Set {model_col} = {value}")
            else:
                debug_info["missing_features"].append(model_col)
                if verbose:
                    print(f"  {Fore.RED}✗ {model_col} not in model features")
    
    # Make prediction
    try:
        # Ensure all numeric columns have numeric dtype
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        
        # Preview the processed dataframe
        if verbose:
            print(f"\n{Fore.CYAN}Final input data (first few columns):")
            print(input_df.iloc[:, :5])
        
        prediction = model.predict(input_df)[0]
        debug_info["prediction"] = prediction
        
        if verbose:
            print(f"\n{Fore.GREEN}✅ Predicted laptop price: {prediction:,.2f} TL")
            
            # Add warning for unrealistic predictions
            if prediction <= 0:
                print(f"{Fore.RED}⚠️ Warning: Prediction is negative or zero, which is unrealistic!")
            elif prediction < 5000:
                print(f"{Fore.RED}⚠️ Warning: Prediction seems too low for a laptop!")
            elif prediction > 250000:
                print(f"{Fore.RED}⚠️ Warning: Prediction seems extremely high!")
        
        return {
            "success": True,
            "prediction": prediction,
            "debug": debug_info
        }
    except Exception as e:
        if verbose:
            print(f"{Fore.RED}❌ Error making prediction: {e}")
        return {
            "success": False,
            "error": str(e),
            "debug": debug_info
        }

def test_standard_configs():
    """Test a set of predefined laptop configurations to check model behavior"""
    print(f"\n{Fore.CYAN}===== Testing Standard Laptop Configurations =====\n")
    
    test_configs = [
        {
            'name': 'Budget Gaming Laptop',
            'config': {
                'brand': 'MONSTER', 
                'işlemci modeli': '12500H',
                'ekran kartı': 'NVIDIA GeForce RTX 3050',
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
                'işlemci modeli': '12700H',
                'ekran kartı': 'NVIDIA GeForce RTX 3060',
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
                'işlemci modeli': '13700H',
                'ekran kartı': 'NVIDIA GeForce RTX 4060',
                'İşletim_Sistemi': 'Windows',
                'ram': 32,
                'ssd': 1024,
                'ekran kartı hafızası': 8,
                'temel işlemci hızı': 2.6,
                'maksimum işlemci hızı': 5.0,
            }
        },
        {
            'name': 'MacBook Pro',
            'config': {
                'brand': 'Apple', 
                'işlemci modeli': 'M2',
                'ekran kartı': 'Apple M2',
                'İşletim_Sistemi': 'Mac Os',
                'ram': 16,
                'ssd': 512,
                'ekran kartı hafızası': 0,  # Integrated
                'temel işlemci hızı': 0,  # Different architecture
                'maksimum işlemci hızı': 0,  # Different architecture
            }
        },
        {
            'name': 'Budget Office Laptop',
            'config': {
                'brand': 'LENOVO', 
                'işlemci modeli': 'i5-1235U',
                'ekran kartı': 'Intel Iris Xe Graphics',
                'İşletim_Sistemi': 'Windows',
                'ram': 8,
                'ssd': 256,
                'ekran kartı hafızası': 0,  # Integrated
                'temel işlemci hızı': 1.3,
                'maksimum işlemci hızı': 4.4,
            }
        }
    ]
    
    results = []
    
    for test in test_configs:
        print(f"{Fore.CYAN}Testing: {test['name']}")
        print(f"{Fore.YELLOW}Configuration:")
        pprint(test['config'])
        
        result = predict_price(test['config'], verbose=False)
        
        if result['success']:
            print(f"{Fore.GREEN}Prediction: {result['prediction']:,.2f} TL")
            status = "Success"
            
            # Check if prediction seems reasonable
            if result['prediction'] <= 0:
                print(f"{Fore.RED}⚠️ Warning: Prediction is negative or zero!")
                status = "Unrealistic (zero/negative)"
            elif result['prediction'] < 5000:
                print(f"{Fore.RED}⚠️ Warning: Prediction seems too low!")
                status = "Unrealistic (too low)"
            elif result['prediction'] > 250000:
                print(f"{Fore.RED}⚠️ Warning: Prediction seems extremely high!")
                status = "Unrealistic (too high)"
        else:
            print(f"{Fore.RED}Error: {result['error']}")
            status = f"Error: {result['error']}"
        
        # Check for encoding issues
        if result['debug']['encoding_issues']:
            print(f"{Fore.YELLOW}Found {len(result['debug']['encoding_issues'])} encoding issues:")
            for issue in result['debug']['encoding_issues']:
                print(f"  {Fore.YELLOW}- {issue['field']}: '{issue['value']}' not recognized")
        
        results.append({
            'name': test['name'],
            'prediction': result.get('prediction', 'Error'),
            'status': status,
            'encoding_issues': len(result['debug']['encoding_issues'])
        })
        
        print("\n" + "-" * 80 + "\n")
    
    # Print summary table
    print(f"{Fore.CYAN}===== Test Summary =====")
    print(f"{'Name':<20} {'Prediction':<15} {'Status':<25} {'Encoding Issues'}")
    print("-" * 70)
    for r in results:
        pred = f"{r['prediction']:,.2f} TL" if isinstance(r['prediction'], (int, float)) else r['prediction']
        print(f"{r['name']:<20} {pred:<15} {r['status']:<25} {r['encoding_issues']}")

if __name__ == "__main__":
    print(f"{Fore.CYAN}===== Laptop Price Prediction Test Tool =====")
    
    # Run basic model info check
    print(f"\n{Fore.YELLOW}Label Encoder Information:")
    for col, encoder in sorted(label_encoders.items()):
        if col in categorical_columns:
            print(f"  {col}: {len(encoder.classes_)} unique values")
    
    # Test a single standard configuration first
    standard_config = {
        'brand': 'MSI',
        'işlemci modeli': '12700H',
        'ekran kartı': 'NVIDIA GeForce RTX 3060',
        'İşletim_Sistemi': 'Windows',
        'ram': 16,
        'ssd': 512,
        'ekran kartı hafızası': 6,
        'temel işlemci hızı': 2.3,
        'maksimum işlemci hızı': 4.7,
    }
    
    print(f"\n{Fore.CYAN}==== Testing Standard Configuration ====")
    predict_price(standard_config)
    
    # Test multiple configurations
    test_standard_configs()
