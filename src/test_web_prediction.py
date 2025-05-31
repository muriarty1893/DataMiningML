#!/usr/bin/env python3
"""
Test the web application with HTTP requests
"""
import requests
import json

def test_prediction():
    url = "http://127.0.0.1:5005/predict"
    
    # Test configuration - Gaming laptop
    data = {
        'brand': 'ASUS',
        'ekran kartı': 'NVIDIA GeForce RTX 3050',
        'işlemci modeli': 'i5-12500H',
        'İşletim_Sistemi': 'Windows',
        'ram': '16',
        'ssd': '512',
        'ekran kartı hafızası': '4',
        'temel işlemci hızı': '2.5',
        'maksimum işlemci hızı': '4.5'
    }
    
    print("🔍 Testing web application prediction...")
    print(f"📊 Input data: {data}")
    print("-" * 50)
    
    try:
        response = requests.post(url, data=data)
        print(f"🌐 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            if result.get('success'):
                price = result.get('prediction')
                print(f"💰 Predicted Price: {price:,.2f} TL")
                print("✅ Prediction successful!")
                
                # Check if prediction is reasonable for this config
                if 20000 <= price <= 40000:
                    print("📊 Price seems reasonable for this configuration")
                else:
                    print("⚠️  Price might be outside expected range")
            else:
                print(f"❌ Prediction failed: {result.get('error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_prediction()
