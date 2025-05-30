# Laptop Fiyat Tahmin Projesi

Bu proje, laptop özelliklerine göre fiyat tahmini yapan bir makine öğrenmesi modeli ve web arayüzü içermektedir.

## Proje Yapısı

```
.
├── data/                   # Veri dosyaları
│   └── doldurulmus_veri.csv
├── docs/                   # Dokümantasyon
│   ├── README.md
│   └── model_comparison_results.csv
├── models/                 # Eğitilmiş modeller ve ilgili dosyalar
│   ├── xgboost_model.joblib
│   ├── label_encoders.joblib
│   ├── categorical_columns.json
│   └── feature_names.json
├── src/                    # Kaynak kodlar
│   ├── app.py
│   └── model_testing.py
├── static/                 # Statik dosyalar (görseller)
│   ├── model_comparison.png
│   ├── best_model_predictions.png
│   ├── error_distribution.png
│   └── *_feature_importance.png
└── templates/             # Web şablonları
    └── index.html
```

## Proje İçeriği

- [Proje Hakkında](#proje-hakkında)
- [Veri Seti](#veri-seti)
- [Model Geliştirme](#model-geliştirme)
- [Web Arayüzü](#web-arayüzü)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Model Performansı](#model-performansı)
- [Sonuçlar ve Değerlendirme](#sonuçlar-ve-değerlendirme)

## Proje Hakkında

Bu proje, laptop özelliklerini kullanarak fiyat tahmini yapan bir makine öğrenmesi modeli geliştirmeyi ve bu modeli kullanıcı dostu bir web arayüzü ile sunmayı amaçlamaktadır. Proje, veri toplama, veri ön işleme, model geliştirme ve web arayüzü geliştirme aşamalarını içermektedir.

## Veri Seti

Projede kullanılan veri seti, çeşitli laptop özelliklerini ve fiyatlarını içermektedir. Veri setinde şu özellikler bulunmaktadır:

- Marka (brand)
- İşlemci modeli
- RAM miktarı
- SSD kapasitesi
- Ekran kartı
- Ekran kartı hafızası
- İşlemci çekirdek sayısı
- Temel işlemci hızı
- Maksimum işlemci hızı
- Ekran yenileme hızı
- İşletim sistemi

## Model Geliştirme

Projede farklı makine öğrenmesi modelleri test edilmiş ve en iyi performans gösteren model seçilmiştir. Test edilen modeller:

1. Linear Regression
2. Random Forest
3. Gradient Boosting
4. SVR (Support Vector Regression)
5. XGBoost

### Model Karşılaştırması

![Model Karşılaştırması](../static/model_comparison.png)

Model performans karşılaştırması sonuçları:

| Model | R2 Score | RMSE | CV R2 Mean |
|-------|----------|------|------------|
| XGBoost | 0.95 | 12,485.58 | 0.86 |
| Random Forest | 0.93 | 14,049.15 | 0.85 |
| Gradient Boosting | 0.92 | 14,687.80 | 0.84 |
| Linear Regression | 0.76 | 26,431.92 | 0.63 |
| SVR | -0.09 | 55,859.70 | -0.15 |

### En İyi Model: XGBoost

XGBoost modeli, en yüksek R2 skoru (0.95) ve en düşük RMSE değeri (12,485.58) ile en iyi performansı göstermiştir.

![XGBoost Tahminleri](../static/best_model_predictions.png)

### Özellik Önemliliği

![XGBoost Özellik Önemliliği](../static/xgboost_feature_importance.png)

## Web Arayüzü

Proje, Flask web framework'ü kullanılarak geliştirilmiş bir web arayüzü içermektedir. Arayüz, kullanıcıların laptop özelliklerini girmesine ve fiyat tahmini almasına olanak sağlar.

### Arayüz Özellikleri

- Kullanıcı dostu form tasarımı
- Kategorik veriler için dropdown menüler
- Sayısal veriler için input alanları
- Anlık fiyat tahmini
- Responsive tasarım

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Flask uygulamasını başlatın:
```bash
cd src
python app.py
```

3. Tarayıcınızda `http://127.0.0.1:5001` adresine gidin.

## Kullanım

1. Web arayüzünde laptop özelliklerini girin
2. Kategorik veriler için dropdown menülerden seçim yapın
3. Sayısal veriler için uygun değerleri girin
4. "Tahmin Et" butonuna tıklayın
5. Tahmin edilen fiyatı görüntüleyin

## Model Performansı

### Tahmin vs Gerçek Değer Grafiği

![Tahmin vs Gerçek](../static/best_model_predictions.png)

### Hata Dağılımı

![Hata Dağılımı](../static/error_distribution.png)

## Sonuçlar ve Değerlendirme

- XGBoost modeli %95 doğruluk oranı ile en iyi performansı göstermiştir
- Cross-validation sonuçları modelin güvenilirliğini doğrulamaktadır
- Hata dağılımı analizi, modelin tutarlı tahminler yaptığını göstermektedir
- Web arayüzü, modelin kullanımını kolaylaştırmaktadır

## Gelecek Geliştirmeler

1. Model performansını artırmak için hiperparametre optimizasyonu
2. Daha fazla özellik ekleme
3. Kullanıcı geri bildirimleri ile model güncelleme
4. Arayüz geliştirmeleri ve yeni özellikler ekleme 