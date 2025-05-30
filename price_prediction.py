import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('doldurulmus_veri.csv')

features = ['İşlemci_Modeli', 'Ram', 'SSD', 'İşlemci_Çekirdek_Sayısı', 
           'Temel_İşlemci_Hızı', 'Ekran_Kartı_Hafızası', 'Maksimum_İşlemci_Hızı',
           'Ekran_Yenileme_Hızı', 'brand', 'Kullanım_Amacı', 'Panel_Tipi']

target = 'price'

le = LabelEncoder()
for feature in ['İşlemci_Modeli', 'brand', 'Kullanım_Amacı', 'Panel_Tipi']:
    df[feature] = le.fit_transform(df[feature].astype(str))

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf'),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2
    }
    
    print(f"\n{name} Sonuçları:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {np.sqrt(mse):.2f}")
    print(f"R2 Score: {r2:.2f}")

best_model = max(results.items(), key=lambda x: x[1]['R2'])
print(f"\nEn iyi model: {best_model[0]}")
print(f"R2 Score: {best_model[1]['R2']:.2f}")

rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Özellik Önemlilikleri')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Fiyat')
plt.ylabel('Tahmin Edilen Fiyat')
plt.title('Tahmin vs Gerçek Değer')
plt.tight_layout()
plt.savefig('prediction_vs_actual.png')
plt.close() 