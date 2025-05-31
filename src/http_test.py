import requests
print("HTTP test başlıyor...")

try:
    response = requests.get('http://127.0.0.1:5004')
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("✅ Web uygulaması çalışıyor!")
    else:
        print("❌ Web uygulaması problemi var")
except Exception as e:
    print(f"❌ Bağlantı hatası: {e}")

print("Test tamamlandı")
