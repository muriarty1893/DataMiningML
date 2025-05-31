#!/usr/bin/env python3
"""
Web uygulaması üzerinden model accuracy testi
"""
import requests
import json

# Test konfigürasyonları
test_configs = [
    {
        'name': 'Budget Office Laptop',
        'data': {
            'brand': 'ACER',
            'ekran kartı': 'Intel UHD Graphics',
            'işlemci modeli': '1005G1',
            'İşletim_Sistemi': 'Windows',
            'ram': '8',
            'ssd': '256',
            'ekran kartı hafızası': '0',
            'temel işlemci hızı': '1.2',
            'maksimum işlemci hızı': '3.4'
        },
        'expected_min': 8000,
        'expected_max': 15000
    },
    {
        'name': 'Gaming Laptop - RTX 3050',
        'data': {
            'brand': 'ASUS',
            'ekran kartı': 'NVIDIA GeForce RTX 3050',
            'işlemci modeli': '12700H',
            'İşletim_Sistemi': 'Windows',
            'ram': '16',
            'ssd': '512',
            'ekran kartı hafızası': '4',
            'temel işlemci hızı': '2.3',
            'maksimum işlemci hızı': '4.7'
        },
        'expected_min': 18000,
        'expected_max': 35000
    },
    {
        'name': 'High-end Gaming Laptop',
        'data': {
            'brand': 'MONSTER',
            'ekran kartı': 'NVIDIA GeForce RTX 4060',
            'işlemci modeli': '13700H',
            'İşletim_Sistemi': 'Windows',
            'ram': '32',
            'ssd': '1024',
            'ekran kartı hafızası': '8',
            'temel işlemci hızı': '2.4',
            'maksimum işlemci hızı': '5.0'
        },
        'expected_min': 35000,
        'expected_max': 60000
    },
    {
        'name': 'MacBook Air M2',
        'data': {
            'brand': 'Apple',
            'ekran kartı': 'Apple M2',
            'işlemci modeli': 'M2',
            'İşletim_Sistemi': 'Mac Os',
            'ram': '16',
            'ssd': '512',
            'ekran kartı hafızası': '0',
            'temel işlemci hızı': '0',
            'maksimum işlemci hızı': '0'
        },
        'expected_min': 25000,
        'expected_max': 45000
    }
]

def test_prediction(config):
    """Tek bir konfigürasyonu test et"""
    url = 'http://127.0.0.1:5004/predict'
    
    try:
        response = requests.post(url, data=config['data'])
        result = response.json()
        
        if result.get('success'):
            prediction = result['prediction']
            expected_min = config['expected_min']
            expected_max = config['expected_max']
            
            is_accurate = expected_min <= prediction <= expected_max
            
            print(f"\n{'='*50}")
            print(f"Test: {config['name']}")
            print(f"{'='*50}")
            print(f"💰 Tahmin: {prediction:,.2f} TL")
            print(f"📊 Beklenen: {expected_min:,} - {expected_max:,} TL")
            
            if is_accurate:
                print("✅ DOĞRU - Tahmin beklenen aralıkta")
                status = "ACCURATE"
            else:
                if prediction < expected_min:
                    error_pct = ((expected_min - prediction) / expected_min) * 100
                    print(f"❌ ÇOK DÜŞÜK - Minimum'dan %{error_pct:.1f} düşük")
                    status = "TOO_LOW"
                else:
                    error_pct = ((prediction - expected_max) / expected_max) * 100
                    print(f"❌ ÇOK YÜKSEK - Maksimum'dan %{error_pct:.1f} yüksek")
                    status = "TOO_HIGH"
            
            # Fuzzy matching kullanıldı mı?
            if result.get('used_mappings'):
                print(f"🔄 Fuzzy Matching Kullanıldı:")
                for field, mapped_value in result['used_mappings'].items():
                    print(f"  • {field}: '{config['data'][field]}' → '{mapped_value}'")
            
            # Öneriler varsa göster
            if result.get('recommendations'):
                print(f"💡 Öneriler:")
                for rec in result['recommendations'][:2]:  # İlk 2 öneriyi göster
                    print(f"  • {rec}")
            
            return {
                'name': config['name'],
                'prediction': prediction,
                'expected_min': expected_min,
                'expected_max': expected_max,
                'is_accurate': is_accurate,
                'status': status,
                'used_mappings': result.get('used_mappings', {})
            }
        else:
            print(f"❌ Hata: {result.get('error', 'Bilinmeyen hata')}")
            return None
            
    except Exception as e:
        print(f"❌ Bağlantı hatası: {e}")
        return None

def main():
    print("🔍 Model Accuracy Testi Başlıyor...")
    print(f"🌐 Test URL: http://127.0.0.1:5004/predict")
    print(f"📊 {len(test_configs)} farklı laptop konfigürasyonu test edilecek")
    
    results = []
    
    for config in test_configs:
        result = test_prediction(config)
        if result:
            results.append(result)
    
    # Özet
    print(f"\n{'='*60}")
    print("📈 MODEL ACCURACY ÖZETİ")
    print(f"{'='*60}")
    
    if results:
        total_tests = len(results)
        accurate_predictions = sum(1 for r in results if r['is_accurate'])
        accuracy_rate = (accurate_predictions / total_tests) * 100
        
        print(f"✅ Doğru Tahminler: {accurate_predictions}/{total_tests}")
        print(f"📊 Genel Doğruluk: %{accuracy_rate:.1f}")
        
        predictions = [r['prediction'] for r in results]
        print(f"💰 Ortalama Tahmin: {sum(predictions)/len(predictions):,.2f} TL")
        print(f"📊 Fiyat Aralığı: {min(predictions):,.2f} - {max(predictions):,.2f} TL")
        
        # Yanlış tahminleri göster
        inaccurate = [r for r in results if not r['is_accurate']]
        if inaccurate:
            print(f"\n⚠️  YANLIŞ TAHMİNLER:")
            for result in inaccurate:
                print(f"  • {result['name']}: {result['prediction']:,.2f} TL ({result['status']})")
        
        # Fuzzy matching kullanımı
        mappings_used = [r for r in results if r['used_mappings']]
        if mappings_used:
            print(f"\n🔄 FUZZY MATCHING KULLANIMI:")
            for result in mappings_used:
                print(f"  • {result['name']}: {len(result['used_mappings'])} alan eşleştirildi")
        
        # Değerlendirme
        print(f"\n💡 DEĞERLENDİRME:")
        if accuracy_rate >= 80:
            print("  ✅ Model performansı mükemmel!")
            print("  💡 Model güvenle kullanılabilir")
        elif accuracy_rate >= 60:
            print("  ⚠️  Model performansı kabul edilebilir ama geliştirilebilir")
            print("  💡 Daha fazla veri ile yeniden eğitim düşünülebilir")
        else:
            print("  ❌ Model performansı yetersiz")
            print("  💡 Model yeniden eğitilmeli")
            print("  💡 Veri kalitesi ve özellik mühendisliği gözden geçirilmeli")
    else:
        print("❌ Hiçbir test başarıyla tamamlanamadı")

if __name__ == "__main__":
    main()
