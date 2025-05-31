from flask import Flask, render_template, request, jsonify
import joblib
import json
import pandas as pd
import numpy as np
import os
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('app')

def fuzzy_match_gpu(value, encoder_classes, threshold=0.7):
    """
    Fuzzy match GPU models using pattern recognition
    
    Args:
        value: The input value to match
        encoder_classes: List of available encoder classes
        threshold: Similarity threshold (0-1)
        
    Returns:
        Matched value or None
    """
    # Normalize input value
    input_value = str(value).strip().lower()
    
    # If exact match exists, return it
    for ec in encoder_classes:
        if str(ec).strip().lower() == input_value:
            return ec
    
    # Create a normalized version for each class
    normalized_classes = {str(ec).strip().lower(): ec for ec in encoder_classes}
    
    # Extract GPU brand and model number from input
    gpu_patterns = [
        # NVIDIA pattern
        r'(?:nvidia|geforce|rtx|gtx)\s*(?:rtx|gtx)?\s*(\d{4,})\s*(?:ti|super)?',
        # AMD pattern
        r'(?:amd|radeon)\s*(?:rx|vega)?\s*(\d{3,})\s*(?:m|xt)?',
        # Intel pattern
        r'(?:intel|iris|uhd)\s*(?:iris|uhd|xe)?\s*(?:graphics)?\s*(?:(\d{3,}))?'
    ]
    
    # Extract model information from input
    model_number = None
    brand = None
    
    # Check if it's NVIDIA
    if re.search(r'nvidia|geforce|rtx|gtx', input_value):
        brand = 'nvidia'
        match = re.search(gpu_patterns[0], input_value)
        if match and match.group(1):
            model_number = match.group(1)
    
    # Check if it's AMD
    elif re.search(r'amd|radeon', input_value):
        brand = 'amd'
        match = re.search(gpu_patterns[1], input_value)
        if match and match.group(1):
            model_number = match.group(1)
    
    # Check if it's Apple/M-series
    elif re.search(r'apple|m\d', input_value):
        brand = 'apple'
        match = re.search(r'(?:apple\s+)?m(\d+)(?:\s+(?:pro|max|ultra))?', input_value)
        if match and match.group(1):
            model_number = f"M{match.group(1)}"
    
    logger.info(f"Extracted GPU info: brand={brand}, model_number={model_number}")
    
    # If we found brand and model, look for matches
    matches = []
    if brand and model_number:
        for normalized, original in normalized_classes.items():
            if brand in normalized and model_number in normalized:
                matches.append((original, 0.9))  # High confidence match
            elif brand in normalized and re.search(rf'\d{{{len(model_number)}}}', normalized):
                # Same brand, different model number but same length
                matches.append((original, 0.6))
    
    # If no brand/model match, try just matching the brand
    if not matches and brand:
        for normalized, original in normalized_classes.items():
            if brand in normalized:
                matches.append((original, 0.5))  # Lower confidence match
    
    # If still no matches, try partial string matching
    if not matches:
        for normalized, original in normalized_classes.items():
            # Check if input contains any part of the class name or vice versa
            if len(input_value) >= 3 and (input_value in normalized or any(word in normalized for word in input_value.split() if len(word) >= 3)):
                matches.append((original, 0.3))  # Very low confidence match
    
    # Sort by confidence
    matches.sort(key=lambda x: x[1], reverse=True)
    
    # Return the best match if it's above threshold
    if matches and matches[0][1] >= threshold:
        logger.info(f"Found match for '{value}': '{matches[0][0]}' with score {matches[0][1]}")
        return matches[0][0]
    
    # If all else fails, return the most common GPU in the encoder
    # Count frequency of each class
    if encoder_classes.size > 0:
        logger.warning(f"No good match found for '{value}'. Using most frequent GPU.")
        values, counts = np.unique(encoder_classes, return_counts=True)
        most_common = values[counts.argmax()]
        return most_common
    
    return None

def fuzzy_match_cpu(value, encoder_classes, threshold=0.7):
    """
    Fuzzy match CPU models using pattern recognition
    
    Args:
        value: The input value to match
        encoder_classes: List of available encoder classes
        threshold: Similarity threshold (0-1)
        
    Returns:
        Matched value or None
    """
    # Normalize input value
    input_value = str(value).strip().lower()
    
    # If exact match exists, return it
    for ec in encoder_classes:
        if str(ec).strip().lower() == input_value:
            return ec
    
    # Create a normalized version for each class
    normalized_classes = {str(ec).strip().lower(): ec for ec in encoder_classes}
    
    # Extract CPU brand, series and model number from input
    cpu_patterns = [
        # Intel pattern
        r'(?:intel|core)?\s*(?:i\d+|celeron|pentium)?\-?(\d{4,})\w*',
        # AMD pattern
        r'(?:amd|ryzen)?\s*(?:ryzen|athlon)?\s*(\d+)?\s*(\d{4,})\w*',
        # Apple pattern
        r'(?:apple|m)\s*(\d+)?(?:\s+(?:pro|max|ultra))?'
    ]
    
    # Extract model information from input
    model_number = None
    brand = None
    
    # Check if it's Intel
    if re.search(r'intel|core|i\d+|celeron|pentium', input_value):
        brand = 'intel'
        match = re.search(cpu_patterns[0], input_value)
        if match and match.group(1):
            model_number = match.group(1)
    
    # Check if it's AMD
    elif re.search(r'amd|ryzen|athlon', input_value):
        brand = 'amd'
        match = re.search(cpu_patterns[1], input_value)
        if match and (match.group(1) or match.group(2)):
            model_number = match.group(2) if match.group(2) else match.group(1)
    
    # Check if it's Apple
    elif re.search(r'apple|m\d', input_value):
        brand = 'apple'
        match = re.search(r'(?:apple\s+)?m(\d+)(?:\s+(?:pro|max|ultra))?', input_value)
        if match and match.group(1):
            model_number = f"M{match.group(1)}"
    
    logger.info(f"Extracted CPU info: brand={brand}, model_number={model_number}")
    
    # If we identified the model number, look for matches
    matches = []
    if model_number:
        for normalized, original in normalized_classes.items():
            if model_number in normalized:
                matches.append((original, 0.8))  # High confidence match for model number
    
    # If no model number matches, try just matching the input directly
    # Look for partial matches (more relaxed)
    if not matches and len(input_value) >= 3:
        for normalized, original in normalized_classes.items():
            if input_value in normalized or normalized in input_value:
                # Calculate string similarity as confidence score
                longer = max(len(input_value), len(normalized))
                if longer == 0:
                    continue
                similarity = (longer - abs(len(input_value) - len(normalized))) / longer
                matches.append((original, similarity * 0.7))  # Scale down the confidence
    
    # Sort by confidence
    matches.sort(key=lambda x: x[1], reverse=True)
    
    # Return the best match if it's above threshold
    if matches and matches[0][1] >= threshold:
        logger.info(f"Found match for '{value}': '{matches[0][0]}' with score {matches[0][1]}")
        return matches[0][0]
    
    # If all else fails, return the most common CPU in the encoder
    if encoder_classes.size > 0:
        logger.warning(f"No good match found for '{value}'. Using most frequent CPU.")
        values, counts = np.unique(encoder_classes, return_counts=True)
        most_common = values[counts.argmax()]
        return most_common
    
    return None

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
    # Gereksiz ve tekrar eden alanları kaldır - ama önemli alanları koru
    features_to_keep = ['İşlemci_Modeli', 'Ekran_Kartı']  # Bu alanları asla kaldırma
    categorical_columns = [c for c in categorical_columns if c not in features_to_remove or c in features_to_keep]
    
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
        
        # Debug için gelen verileri logla
        print(f"Input data received: {data}")
        print(f"Categorical columns: {categorical_columns}")
        print(f"Available label encoders: {list(label_encoders.keys())}")
                
        # Veriyi DataFrame'e çevir
        input_data = pd.DataFrame([data])
        
        # Feature mapping - UI'daki field adlarını model feature adlarına dönüştür
        feature_mapping = {
            'brand': 'brand',
            'ekran kartı': 'Ekran_Kartı',
            'işlemci modeli': 'İşlemci_Modeli',
            'ekran kartı hafızası': 'Ekran_Kartı_Hafızası',
            'temel işlemci hızı': 'Temel_İşlemci_Hızı',
            'maksimum işlemci hızı': 'Maksimum_İşlemci_Hızı',
            'İşletim_Sistemi': 'İşletim_Sistemi',
            'ram': 'Ram',
            'ssd': 'SSD',
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
        
        # Alternative names that might be used (lowercase, no underscores, etc.)
        alternative_mappings = {
            'ekran boyutu': 'Ekran_Boyutu',
            'çözünürlük': 'Çözünürlük',
            'çözünürlük standartı': 'Çözünürlük_Standartı',
            'panel tipi': 'Panel_Tipi', 
            'ekran yenileme hızı': 'Ekran_Yenileme_Hızı',
            'ekran kartı tipi': 'Ekran_Kartı_Tipi',
            'işlemci çekirdek sayısı': 'İşlemci_Çekirdek_Sayısı',
            'kullanım amacı': 'Kullanım_Amacı',
            'parmak izi okuyucu': 'Parmak_İzi_Okuyucu'
        }
        
        # Add alternative mappings to the main mapping
        feature_mapping.update(alternative_mappings)
        
        # Feature mapping'i uygula
        for ui_field, model_field in feature_mapping.items():
            if ui_field in input_data.columns:
                input_data[model_field] = input_data[ui_field]
        
        # UI field'larını kaldır (sadece model field'ları kalsın)
        ui_fields_to_remove = ['ekran kartı', 'işlemci modeli', 'ekran kartı hafızası', 
                              'temel işlemci hızı', 'maksimum işlemci hızı', 'ram', 'ssd']
        for field in ui_fields_to_remove:
            if field in input_data.columns:
                input_data = input_data.drop(columns=[field])
        
        print(f"After feature mapping and cleanup - columns: {list(input_data.columns)}")
        print(f"Categorical columns to encode: {categorical_columns}")
                
        # Kategorik değişkenleri encode et
        for column in categorical_columns:
            if column in input_data.columns and input_data[column].iloc[0] is not None:
                try:
                    value = input_data[column].iloc[0]
                    print(f"Processing {column} = '{value}'")
                    
                    # Değer encoder'da var mı kontrol et
                    if column in label_encoders:
                        # Exact match
                        if value in label_encoders[column].classes_:
                            encoded_value = label_encoders[column].transform([value])[0]
                            input_data[column] = encoded_value
                            logger.info(f"Exact match found for {column}: {value}")
                        else:
                            logger.warning(f"Value '{value}' not found in encoder for {column}")
                            
                            # For GPU and CPU models, use fuzzy matching
                            if column == 'Ekran_Kartı':
                                match = fuzzy_match_gpu(value, label_encoders[column].classes_)
                                if match:
                                    encoded_value = label_encoders[column].transform([match])[0]
                                    input_data[column] = encoded_value
                                    logger.info(f"Fuzzy match for GPU: '{value}' → '{match}'")
                                else:
                                    input_data[column] = 0
                            elif column == 'İşlemci_Modeli':
                                match = fuzzy_match_cpu(value, label_encoders[column].classes_)
                                if match:
                                    encoded_value = label_encoders[column].transform([match])[0]
                                    input_data[column] = encoded_value
                                    logger.info(f"Fuzzy match for CPU: '{value}' → '{match}'")
                                else:
                                    input_data[column] = 0
                            else:
                                # Try case-insensitive and spacing-insensitive matching for other fields
                                found = False
                                for encoder_value in label_encoders[column].classes_:
                                    if str(value).strip().lower() == str(encoder_value).strip().lower():
                                        encoded_value = label_encoders[column].transform([encoder_value])[0]
                                        input_data[column] = encoded_value
                                        logger.info(f"Case-insensitive match: '{encoder_value}' for '{value}'")
                                        found = True
                                        break
                                
                                # If still not found, use the most frequent value for that column
                                if not found:
                                    values, counts = np.unique(label_encoders[column].classes_, return_counts=True)
                                    most_common = values[counts.argmax()]
                                    encoded_value = label_encoders[column].transform([most_common])[0]
                                    input_data[column] = encoded_value
                                    logger.warning(f"No match for '{value}' in {column}, using most common: '{most_common}'")
                except Exception as e:
                    print(f"Error encoding {column}: {e}")
                    input_data[column] = 0
        
        # Tüm kategorik alanları numeric'e dönüştür (XGBoost için gerekli)
        for column in categorical_columns:
            if column in input_data.columns:
                input_data[column] = pd.to_numeric(input_data[column], errors='coerce')
        
        # Sayısal alanları float'a dönüştür
        for column in numeric_features:
            if column in input_data.columns:
                input_data[column] = pd.to_numeric(input_data[column], errors='coerce')
        
        # Eksik değerleri 0 ile doldur
        input_data = input_data.fillna(0)
        
        # Tüm kolonları float tipine dönüştür (XGBoost için)
        for column in input_data.columns:
            input_data[column] = input_data[column].astype(float)
        
        # Modelin beklediği tüm feature'ları ekle
        model_features = model.get_booster().feature_names
        missing_features = set(model_features) - set(input_data.columns)
        for feature in missing_features:
            input_data[feature] = 0
            print(f"Added missing feature: {feature}")
        
        # Feature'ları modelin beklediği sırayla düzenle
        input_data = input_data[model_features]
        
        # Debug için model input verilerini logla
        print(f"Model input data (first 5 columns): {input_data.iloc[:, :5]}")
        
        # Tahmin yap
        prediction = model.predict(input_data)[0]
        logger.info(f"Prediction result: {prediction}")
        
        # Prediction değerinin makul olup olmadığını kontrol et
        prediction_status = "normal"
        warning_message = None
        
        if prediction <= 0:
            warning_message = "Tahmin edilen fiyat sıfır veya negatif. Bu genellikle giriş bilgilerinin model tarafından tanınmadığını gösterir."
            prediction_status = "unrealistic_low"
            logger.warning(f"Unrealistic prediction: {prediction} (negative/zero)")
        elif prediction < 5000:
            warning_message = "Tahmin edilen fiyat çok düşük. Lütfen girdiğiniz bilgileri kontrol edin."
            prediction_status = "likely_low"
            logger.warning(f"Likely too low prediction: {prediction}")
        elif prediction > 250000:
            warning_message = "Tahmin edilen fiyat çok yüksek. Lütfen girdiğiniz bilgileri kontrol edin."
            prediction_status = "likely_high"
            logger.warning(f"Likely too high prediction: {prediction}")
        
        # Collect any potential issues with the input data
        data_issues = []
        used_mappings = {}
        
        for column in categorical_columns:
            if column in input_data.columns:
                ui_fields = [key for key, val in feature_mapping.items() if val == column]
                ui_field = ui_fields[0] if ui_fields else column
                
                # Check if a fuzzy mapping was used
                original_value = data.get(ui_field, None) if ui_field in data else None
                
                if original_value and original_value not in label_encoders.get(column, {}).classes_:
                    # There was a value that needed fuzzy matching
                    encoded_value = input_data[column].iloc[0]
                    # Find which value this corresponds to
                    if column in label_encoders:
                        for i, cls in enumerate(label_encoders[column].classes_):
                            if label_encoders[column].transform([cls])[0] == encoded_value:
                                data_issues.append({
                                    "field": ui_field,
                                    "original": original_value,
                                    "mapped_to": cls
                                })
                                used_mappings[ui_field] = cls
                                break
        
        # Generate recommendations based on configuration
        recommendations = []
        
        # Check RAM vs SSD balance
        ram_value = data.get('ram', 0)
        ssd_value = data.get('ssd', 0)
        
        try:
            ram_value = float(ram_value) if ram_value else 0
            ssd_value = float(ssd_value) if ssd_value else 0
            
            if ram_value > 32 and ssd_value < 512:
                recommendations.append("Yüksek RAM ile daha büyük bir SSD (en az 512GB) kullanmanız önerilir.")
            elif ram_value < 8 and ssd_value > 1024:
                recommendations.append("Yüksek depolama alanına sahip bir bilgisayar için daha fazla RAM (en az 16GB) düşünebilirsiniz.")
        
            # Check GPU and CPU balance
            gpu_value = data.get('ekran kartı', '')
            cpu_value = data.get('işlemci modeli', '')
            
            if gpu_value and ('rtx 30' in str(gpu_value).lower() or 'rtx 40' in str(gpu_value).lower()):
                if ram_value < 16:
                    recommendations.append("Güçlü ekran kartınız için daha fazla RAM (en az 16GB) önerilir.")
            
            # Price range recommendation
            if 10000 <= prediction <= 15000:
                recommendations.append("Bu fiyat aralığında genellikle temel ofis ve günlük kullanım bilgisayarları bulunur.")
            elif 15001 <= prediction <= 25000:
                recommendations.append("Bu fiyat aralığında orta seviye oyun ve multimedya bilgisayarları bulunur.")
            elif 25001 <= prediction <= 40000:
                recommendations.append("Bu fiyat aralığında yüksek performanslı oyun ve içerik üretimi bilgisayarları bulunur.")
            elif prediction > 40000:
                recommendations.append("Premium fiyat aralığında profesyonel kullanıma yönelik bilgisayarlar bulunur.")
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'status': prediction_status,
            'warning': warning_message,
            'data_issues': data_issues,
            'used_mappings': used_mappings,
            'recommendations': recommendations
        })
    
    except Exception as e:
        import traceback
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5005)