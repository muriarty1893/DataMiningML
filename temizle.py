import pandas as pd
import re

# Temizleme fonksiyonu
units_pattern = re.compile(r'\s?(gb|mb|tb|ghz|hz)', re.IGNORECASE)

def clean_and_convert(value):
    if isinstance(value, str):
        value = units_pattern.sub('', value)
        value = value.replace(',', '.')
        try:
            return float(value)
        except ValueError:
            return value
    return value

# CSV dosyasını oku
input_file = "GBsilinecek.csv"
output_file = "GBsilinecek_cleaned.csv"

df = pd.read_csv(input_file)
df_cleaned = df.applymap(clean_and_convert)
df_cleaned.to_csv(output_file, index=False)

print(f"Temizlenmiş dosya kaydedildi: {output_file}")
