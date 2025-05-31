#!/usr/bin/env python3
# Model compatibility script for fixing version issues
import joblib
import json
import os
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def update_model_format():
    """Update the model to the latest format to avoid version warnings."""
    model_path = os.path.join('..', 'models', 'xgboost_model.joblib')
    backup_path = os.path.join('..', 'models', 'xgboost_model.joblib.bak')
    
    print(f"Loading model from {model_path}...")
    try:
        # Create backup
        if not os.path.exists(backup_path):
            print(f"Creating backup at {backup_path}...")
            with open(model_path, 'rb') as src, open(backup_path, 'wb') as dst:
                dst.write(src.read())
        
        # Load model
        model = joblib.load(model_path)
        
        # Save model in new format
        temp_path = os.path.join('..', 'models', 'model.json')
        print(f"Saving model in new format to {temp_path}...")
        model.save_model(temp_path)
        
        # Load model back in new format
        print("Loading model in new format...")
        updated_model = xgb.XGBRegressor()
        updated_model.load_model(temp_path)
        
        # Save model back to original path
        print(f"Saving updated model back to {model_path}...")
        joblib.dump(updated_model, model_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        print("Model format update completed successfully.")
        return True
        
    except Exception as e:
        print(f"Error updating model format: {e}")
        return False

def update_label_encoders():
    """Update the label encoders to the latest format to avoid version warnings."""
    encoders_path = os.path.join('..', 'models', 'label_encoders.joblib')
    backup_path = os.path.join('..', 'models', 'label_encoders.joblib.bak')
    
    print(f"Loading label encoders from {encoders_path}...")
    try:
        # Create backup
        if not os.path.exists(backup_path):
            print(f"Creating backup at {backup_path}...")
            with open(encoders_path, 'rb') as src, open(backup_path, 'wb') as dst:
                dst.write(src.read())
        
        # Load encoders
        label_encoders = joblib.load(encoders_path)
        
        # Create new encoders with the same classes
        updated_encoders = {}
        for key, encoder in label_encoders.items():
            new_encoder = LabelEncoder()
            new_encoder.classes_ = encoder.classes_
            updated_encoders[key] = new_encoder
        
        # Save updated encoders
        print(f"Saving updated label encoders to {encoders_path}...")
        joblib.dump(updated_encoders, encoders_path)
        
        print("Label encoders update completed successfully.")
        return True
        
    except Exception as e:
        print(f"Error updating label encoders: {e}")
        return False

def validate_mappings():
    """Validate that all feature mappings are correct."""
    label_encoders_path = os.path.join('..', 'models', 'label_encoders.joblib')
    feature_names_path = os.path.join('..', 'models', 'feature_names.json')
    categorical_columns_path = os.path.join('..', 'models', 'categorical_columns.json')
    
    print("Validating feature mappings...")
    try:
        # Load data
        label_encoders = joblib.load(label_encoders_path)
        
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
            
        with open(categorical_columns_path, 'r') as f:
            categorical_columns = json.load(f)
        
        # Check that all categorical columns are in label encoders
        missing_encoders = [col for col in categorical_columns if col not in label_encoders]
        if missing_encoders:
            print(f"Warning: The following categorical columns don't have label encoders: {missing_encoders}")
        
        # Check that all label encoders are in feature names
        missing_features = [col for col in label_encoders.keys() if col not in feature_names]
        if missing_features:
            print(f"Warning: The following encoder columns are not in feature names: {missing_features}")
            
        # Print encoder info
        print("\nLabel encoder information:")
        for col, encoder in label_encoders.items():
            print(f"  {col}: {len(encoder.classes_)} unique values")
            
        print("\nValidation completed.")
        return True
        
    except Exception as e:
        print(f"Error validating mappings: {e}")
        return False
        
if __name__ == "__main__":
    print("Starting model compatibility fixes...")
    
    # Run all fixes
    update_model_format()
    update_label_encoders()
    validate_mappings()
    
    print("\nAll fixes completed. Please restart your application.")
