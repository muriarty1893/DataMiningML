import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import os

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Veri setini yükle
data_path = os.path.join('..', 'data', 'doldurulmus_veri.csv')
df = pd.read_csv(data_path)

# Kategorik değişkenleri encode et
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Feature ve target değişkenlerini ayır
X = df.drop('price', axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelleri tanımla
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf'),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
}

# Sonuçları saklamak için dictionary
results = {}

# Her model için detaylı analiz yap
for name, model in models.items():
    logging.info(f"\n{name} Analizi:")
    
    # Modeli eğit
    model.fit(X_train, y_train)
    
    # Tahminler
    y_pred = model.predict(X_test)
    
    # Temel metrikler
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    # Sonuçları sakla
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std(),
        'y_pred': y_pred,
        'y_test': y_test
    }
    
    # Metrikleri yazdır
    logging.info(f"MSE: {mse:.2f}")
    logging.info(f"RMSE: {rmse:.2f}")
    logging.info(f"R2 Score: {r2:.2f}")
    logging.info(f"Cross-validation R2 scores: {cv_scores}")
    logging.info(f"Cross-validation R2 mean: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    
    # Özellik önemliliklerini göster (eğer model destekliyorsa)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
        plt.title(f'{name} - Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join('..', 'static', f'{name.lower().replace(" ", "_")}_feature_importance.png'))
        plt.close()

# En iyi modeli bul
best_model = max(results.items(), key=lambda x: x[1]['R2'])
logging.info(f"\nEn iyi model: {best_model[0]}")
logging.info(f"R2 Score: {best_model[1]['R2']:.2f}")

# Model karşılaştırma grafiği
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
r2_scores = [results[model]['R2'] for model in model_names]
cv_means = [results[model]['CV_Mean'] for model in model_names]
cv_stds = [results[model]['CV_Std'] for model in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, r2_scores, width, label='Test R2')
plt.bar(x + width/2, cv_means, width, yerr=cv_stds, label='CV R2 Mean')
plt.xlabel('Models')
plt.ylabel('R2 Score')
plt.title('Model Comparison')
plt.xticks(x, model_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join('..', 'static', 'model_comparison.png'))
plt.close()

# Tahmin vs Gerçek değer grafiği (en iyi model için)
best_model_name = best_model[0]
y_pred = results[best_model_name]['y_pred']
y_test = results[best_model_name]['y_test']

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahminler')
plt.title(f'{best_model_name} - Tahmin vs Gerçek')
plt.tight_layout()
plt.savefig(os.path.join('..', 'static', 'best_model_predictions.png'))
plt.close()

# Hata dağılımı analizi
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True)
plt.xlabel('Tahmin Hatası')
plt.ylabel('Frekans')
plt.title(f'{best_model_name} - Hata Dağılımı')
plt.tight_layout()
plt.savefig(os.path.join('..', 'static', 'error_distribution.png'))
plt.close()

# Sonuçları CSV'ye kaydet
results_df = pd.DataFrame({
    'Model': model_names,
    'R2_Score': r2_scores,
    'CV_R2_Mean': cv_means,
    'CV_R2_Std': cv_stds,
    'RMSE': [results[model]['RMSE'] for model in model_names]
})
results_df.to_csv(os.path.join('..', 'docs', 'model_comparison_results.csv'), index=False)

logging.info("\nDetaylı analiz sonuçları 'docs/model_comparison_results.csv' dosyasına kaydedildi.")
logging.info("Görselleştirmeler kaydedildi:")
logging.info("- static/model_comparison.png")
logging.info("- static/best_model_predictions.png")
logging.info("- static/error_distribution.png")
logging.info("- static/*_feature_importance.png") 