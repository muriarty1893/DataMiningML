import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Veriyi oku
df = pd.read_csv("GBsilinecek_cleaned.csv")

# Meta verileri sakla
meta_data = df[["name", "url", "id"]].copy()

# Hedef sütunları hariç tut
df = df.drop(columns=["name", "url", "id"])  # name ve url benzersiz metin, id zaten unique

# Sütunları ayır
num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Sayısal veriler için iteratif doldurma (RandomForestRegressor)
num_imputer = IterativeImputer(estimator=RandomForestRegressor(), random_state=42)

# Kategorik veriler için özel doldurma fonksiyonu
def fill_categorical_with_model(df, cat_cols):
    df_copy = df.copy()
    for col in cat_cols:
        if df_copy[col].isnull().sum() == 0:
            continue
        df_known = df_copy[df_copy[col].notnull()]
        df_missing = df_copy[df_copy[col].isnull()]
        if df_known.shape[0] < 10 or df_missing.shape[0] == 0:
            continue
        # Encode kategorik değişkenler
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_known = df_known.drop(columns=[col])
        y_known = df_known[col]
        X_missing = df_missing.drop(columns=[col])

        X_all = pd.concat([X_known, X_missing])
        X_all_encoded = X_all.copy()
        for c in cat_cols:
            if c in X_all_encoded.columns:
                X_all_encoded[c] = enc.fit_transform(X_all_encoded[[c]])

        X_known_encoded = X_all_encoded.loc[X_known.index]
        X_missing_encoded = X_all_encoded.loc[X_missing.index]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_known_encoded, y_known)
        preds = model.predict(X_missing_encoded)

        df_copy.loc[df_missing.index, col] = preds
    return df_copy

# Sayısal eksikleri doldur
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Kategorik eksikleri doldur
df = fill_categorical_with_model(df, cat_cols)

# Kontrol: hâlâ eksik veri var mı?
missing_after = df.isnull().sum().sort_values(ascending=False)

print("\nEksik veri kontrolü:")
print(missing_after[missing_after > 0])

# Meta verileri geri ekle
df = pd.concat([meta_data, df], axis=1)

# Sonuçları kaydet
df.to_csv("doldurulmus_veri.csv", index=False)
print("\nDoldurulmuş veri 'doldurulmus_veri.csv' dosyasına kaydedildi.")
