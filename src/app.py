from flask import Flask, render_template, request, jsonify
import joblib
import json
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Model ve gerekli dosyaları yükle
model_path = os.path.join('..', 'models', 'xgboost_model.joblib')
label_encoders_path = os.path.join('..', 'models', 'label_encoders.joblib')
feature_names_path = os.path.join('..', 'models', 'feature_names.json')
categorical_columns_path = os.path.join('..', 'models', 'categorical_columns.json')

model = joblib.load(model_path)
label_encoders = joblib.load(label_encoders_path)

with open(feature_names_path, 'r') as f:
    all_features = json.load(f)
    # Gereksiz ve tekrar eden alanları kaldır
    features_to_remove = [
        'name', 'id', 'url', 'ekran kartı tipi',
        'İşlemci_Modeli',  # işlemci modeli yerine
        'Ekran_Kartı',    # ekran kartı yerine
        'Ekran_Kartı_Hafızası',  # ekran kartı hafızası yerine
        'Maksimum_İşlemci_Hızı',  # maksimum işlemci hızı yerine
        'Ram',  # ram yerine
        'SSD',  # ssd yerine
        'Temel_İşlemci_Hızı',  # temel işlemci hızı yerine
        'işletim sistemi'  # İşletim_Sistemi yerine
    ]
    feature_names = [f for f in all_features if f not in features_to_remove]

# İstenen sıralama
ordered_features = [
    'brand',
    'ekran kartı',
    'işlemci modeli',
    'ekran kartı hafızası',
    'temel işlemci hızı',
    'maksimum işlemci hızı',
    'ram',
    'ssd',
    'İşletim_Sistemi'
]

# Diğer feature'ları bul
other_features = [f for f in feature_names if f not in ordered_features]

# Son sıralama: ordered_features + other_features
final_feature_names = ordered_features + other_features

# Sayısal alanların listesi
numeric_features = [
    'ekran kartı hafızası',
    'temel işlemci hızı',
    'maksimum işlemci hızı',
    'ram',
    'ssd',
    'Ekran_Yenileme_Hızı',
    'İşlemci_Çekirdek_Sayısı'
]

with open(categorical_columns_path, 'r') as f:
    categorical_columns = json.load(f)
    # Gereksiz ve tekrar eden alanları kaldır
    categorical_columns = [c for c in categorical_columns if c not in features_to_remove]
    
    # Sadece label_encoders'da var olan kategorik alanları kullan
    categorical_columns = [col for col in categorical_columns if col in label_encoders]

@app.route('/')
def home():
    # Extract unique values for dropdown menus for features not in label_encoders
    additional_categorical_features = [
        'ekran kartı',
        'işlemci modeli',
        'Ekran_Boyutu',
        'Çözünürlük',
        'Çözünürlük_Standartı',
        'Panel_Tipi',
        'Kullanım_Amacı'
    ]
    
    # Create dictionary for dropdown values
    dropdown_values = {}
    
    # Add label encoder values to dropdown_values
    for col in categorical_columns:
        if col in label_encoders:
            dropdown_values[col] = sorted(label_encoders[col].classes_.tolist())
    
    # Create mappings for lowercase/uppercase & Turkish character variations
    feature_mapping = {
        'ekran kartı': 'Ekran_Kartı',
        'işlemci modeli': 'İşlemci_Modeli',
        'ekran kartı hafızası': 'Ekran_Kartı_Hafızası',
        'temel işlemci hızı': 'Temel_İşlemci_Hızı',
        'maksimum işlemci hızı': 'Maksimum_İşlemci_Hızı',
        'İşletim_Sistemi': 'İşletim_Sistemi',
        'brand': 'brand',
        'Kullanım_Amacı': 'Kullanım_Amacı',
        'Ekran_Boyutu': 'Ekran_Boyutu',
        'Çözünürlük': 'Çözünürlük',
        'Çözünürlük_Standartı': 'Çözünürlük_Standartı',
        'Panel_Tipi': 'Panel_Tipi',
        'Ekran_Kartı_Tipi': 'Ekran_Kartı_Tipi',
        'Parmak_İzi_Okuyucu': 'Parmak_İzi_Okuyucu'
    }
    
    # Populate dropdown values from label encoders
    for display_feature, encoder_feature in feature_mapping.items():
        if encoder_feature in label_encoders:
            dropdown_values[display_feature] = sorted(label_encoders[encoder_feature].classes_.tolist())
    
    # Additional feature values could be added here from training data if needed
    # For now we'll use the label encoders as our source for dropdown values
    
    return render_template('index.html', 
                         feature_names=final_feature_names,
                         categorical_columns=categorical_columns + additional_categorical_features,
                         numeric_features=numeric_features,
                         label_encoders=label_encoders,
                         dropdown_values=dropdown_values)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form verilerini al
        data = request.form.to_dict()
        
        # Temel özelliklerin boş olup olmadığını kontrol et
        required_fields = ['brand', 'ekran kartı', 'işlemci modeli', 'İşletim_Sistemi']
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Zorunlu alanlar boş bırakılamaz: {", ".join(missing_fields)}'
            })
        
        # Boş değerleri None olarak ayarla
        for key in data:
            if data[key] == '':
                data[key] = None
        
        # Veriyi DataFrame'e çevir
        input_data = pd.DataFrame([data])
        
        # Kategorik değişkenleri encode et
        for column in categorical_columns:
            if column in input_data.columns and input_data[column].iloc[0] is not None:
                input_data[column] = label_encoders[column].transform([input_data[column].iloc[0]])
        
        # Sayısal alanları float'a dönüştür
        for column in numeric_features:
            if column in input_data.columns:
                input_data[column] = pd.to_numeric(input_data[column], errors='coerce')
        
        # Eksik değerleri 0 ile doldur
        input_data = input_data.fillna(0)
        
        # Modelin beklediği tüm feature'ları ekle
        model_features = model.get_booster().feature_names
        for feature in model_features:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        # Feature'ları modelin beklediği sırayla düzenle
        input_data = input_data[model_features]
        
        # Tahmin yap
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5003) 