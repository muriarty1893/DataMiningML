import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import json

# Veriyi yükle
df = pd.read_csv('doldurulmus_veri.csv')

# Kategorik değişkenleri belirle ve encode et
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Label encoder'ları kaydet
joblib.dump(label_encoders, 'label_encoders.joblib')

# Feature ve target değişkenlerini ayır
X = df.drop('price', axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost modelini eğit
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Modeli kaydet
joblib.dump(model, 'xgboost_model.joblib')

# Feature isimlerini kaydet
feature_names = list(X.columns)
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)

# Kategorik kolonları kaydet
categorical_columns_list = list(categorical_columns)
with open('categorical_columns.json', 'w') as f:
    json.dump(categorical_columns_list, f)

print("Model eğitimi tamamlandı ve kaydedildi.") 