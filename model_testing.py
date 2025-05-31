import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(f'model_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

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
    
    logging.info(f"\n{name} Sonuçları:")
    logging.info(f"MSE: {mse:.2f}")
    logging.info(f"RMSE: {np.sqrt(mse):.2f}")
    logging.info(f"R2 Score: {r2:.2f}")

logging.info("\nCross-Validation Results (R2 Score, 5 folds):")
cv_results = {}
for name, model in models.items():
    # Use the scaled training data for cross-validation
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_results[name] = {
        'CV_Mean_R2': np.mean(scores),
        'CV_Std_R2': np.std(scores)
    }
    logging.info(f"{name}: Mean R2 = {cv_results[name]['CV_Mean_R2']:.2f}, Std R2 = {cv_results[name]['CV_Std_R2']:.2f}")

best_model = max(results.items(), key=lambda x: x[1]['R2'])
logging.info(f"\nEn iyi model: {best_model[0]}")
logging.info(f"R2 Score: {best_model[1]['R2']:.2f}")

# Get predictions from the best model for plotting
best_model_name = best_model[0]
best_model_object = models[best_model_name]
best_model_predictions = best_model_object.predict(X_test_scaled)

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
plt.scatter(y_test, best_model_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Fiyat')
plt.ylabel('Tahmin Edilen Fiyat')
plt.title('Tahmin vs Gerçek Değer')
plt.tight_layout()
plt.savefig('prediction_vs_actual.png')
plt.close()

# Add Residual Plot
plt.figure(figsize=(10, 6))
residuals = y_test - best_model_predictions
plt.scatter(best_model_predictions, residuals, alpha=0.5)
plt.hlines(0, plt.xlim()[0], plt.xlim()[1], colors='r', linestyles='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot (Best Model)')
plt.tight_layout()
plt.savefig('residual_plot.png')
plt.close()

# Add Model R2 Comparison Plot
model_names = list(results.keys())
r2_scores = [results[name]['R2'] for name in model_names]

plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=r2_scores)
plt.ylabel('R2 Score')
plt.title('Model R2 Score Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('model_r2_comparison.png')
plt.close()

# --- Make a prediction based on user input ---
logging.info("\n--- Kullanıcı Girişi ile Fiyat Tahmini ---")

def get_user_input(feature_list):
    user_data = {}
    print("Lütfen aşağıdaki özellikler için değerleri girin:")
    for feature in feature_list:
        user_data[feature] = input(f"{feature}: ")
    return user_data

# Get input from user
user_input_raw = get_user_input(features)

# Create a DataFrame from user input
# Convert single row dict to DataFrame
user_df = pd.DataFrame([user_input_raw])

# Apply the same preprocessing as used for training data
# Identify categorical and numerical features again based on the original list and le usage
categorical_features = ['İşlemci_Modeli', 'brand', 'Kullanım_Amacı', 'Panel_Tipi']
numerical_features = [f for f in features if f not in categorical_features]

# Convert numerical inputs to appropriate types (float/int)
for col in numerical_features:
    try:
        # Attempt to convert to integer first if no decimal is expected
        user_df[col] = pd.to_numeric(user_df[col], errors='coerce')
    except ValueError:
        user_df[col] = np.nan # Handle cases where conversion fails

# Apply Label Encoding to categorical features using the *fitted* le object
for feature in categorical_features:
    # Handle potential new categories not seen during training if necessary
    # For simplicity here, we assume user input categories exist in training data labels
    # A more robust approach might handle unknown labels (e.g., assign a default or use handle_unknown='ignore' if supported)
    try:
        user_df[feature] = le.transform(user_df[feature].astype(str))
    except ValueError as e:
        logging.error(f"Label Encoding error for feature {feature}: {e}. Ensure input category exists in training data.")
        # Decide how to handle this - for now, let's stop or assign a default/NaN
        sys.exit(f"Hata: '{user_input_raw[feature]}' değeri '{feature}' özelliği için geçerli değil.")

# Apply Scaling to numerical features using the *fitted* scaler object
# Ensure numerical columns are indeed numeric before scaling
user_df_numerical = user_df[numerical_features]

# Check for any non-numeric values that failed conversion
if user_df_numerical.isnull().values.any():
     sys.exit("Hata: Sayısal değer girilmesi gereken bir özellik için geçersiz giriş yaptınız.")

user_data_scaled = scaler.transform(user_df_numerical)

# Combine scaled numerical and encoded categorical data
# Ensure the order of columns matches the training data
user_data_processed = pd.DataFrame(user_data_scaled, columns=numerical_features)

# Add back categorical columns (already encoded)
for feature in categorical_features:
     user_data_processed[feature] = user_df[feature]

# Reorder columns to match the training features list
user_data_processed = user_data_processed[features]

# Make prediction using the best model
# Ensure the best_model_object is accessible here. It was defined earlier.
if 'best_model_object' in locals():
    predicted_price = best_model_object.predict(user_data_processed)
    logging.info(f"Tahmin Edilen Fiyat: {predicted_price[0]:.2f}")
else:
    logging.error("Best model object not found. Cannot make prediction.") 