import joblib
import json
import numpy as np

# Model ve encoderları yükle
print("Model yükleniyor...")
model = joblib.load('../models/xgboost_model.joblib')
label_encoders = joblib.load('../models/label_encoders.joblib')

print("✅ Model başarıyla yüklendi")
print(f"Model tipi: {type(model)}")

# Model özelliklerini kontrol et
features = model.get_booster().feature_names
print(f"Model {len(features)} özellik bekliyor")

# Label encoderları kontrol et
print(f"\nKategorik özellikler:")
for col, encoder in label_encoders.items():
    print(f"  {col}: {len(encoder.classes_)} farklı değer")

print("\nModel testi tamamlandı!")
