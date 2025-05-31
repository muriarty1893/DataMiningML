#!/usr/bin/env python3
"""
Test multiple configurations with the web application
"""
import requests
import json

def test_configuration(name, data, expected_range=None):
    url = "http://127.0.0.1:5005/predict"
    
    print(f"\n🔍 Testing: {name}")
    print(f"📊 Input: {data}")
    print("-" * 50)
    
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                price = result.get('prediction')
                print(f"💰 Predicted Price: {price:,.2f} TL")
                
                if expected_range:
                    if expected_range[0] <= price <= expected_range[1]:
                        print(f"✅ ACCURATE - Within expected range ({expected_range[0]:,} - {expected_range[1]:,} TL)")
                    else:
                        print(f"⚠️  Outside expected range ({expected_range[0]:,} - {expected_range[1]:,} TL)")
                else:
                    print("✅ Prediction successful")
                
                # Show mappings if any
                if result.get('used_mappings'):
                    print(f"🔄 Fuzzy matches: {result['used_mappings']}")
                
                return price
            else:
                print(f"❌ Error: {result.get('error')}")
                return None
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def main():
    print("=== WEB APPLICATION COMPREHENSIVE TEST ===")
    
    test_configs = [
        {
            'name': 'Budget Office Laptop',
            'data': {
                'brand': 'ACER',
                'ekran kartı': 'Intel UHD Graphics',
                'işlemci modeli': 'i3-1005G1',
                'İşletim_Sistemi': 'Windows',
                'ram': '8',
                'ssd': '256'
            },
            'expected': (8000, 15000)
        },
        {
            'name': 'Gaming Laptop - RTX 3050',
            'data': {
                'brand': 'ASUS',
                'ekran kartı': 'NVIDIA GeForce RTX 3050',
                'işlemci modeli': 'i5-12500H',
                'İşletim_Sistemi': 'Windows',
                'ram': '16',
                'ssd': '512',
                'ekran kartı hafızası': '4'
            },
            'expected': (20000, 35000)
        },
        {
            'name': 'High-end Gaming Laptop',
            'data': {
                'brand': 'MONSTER',
                'ekran kartı': 'NVIDIA GeForce RTX 4060',
                'işlemci modeli': 'i7-13700H',
                'İşletim_Sistemi': 'Windows',
                'ram': '32',
                'ssd': '1024',
                'ekran kartı hafızası': '8'
            },
            'expected': (40000, 70000)
        },
        {
            'name': 'Apple MacBook',
            'data': {
                'brand': 'Apple',
                'ekran kartı': 'Apple M2',
                'işlemci modeli': 'Apple M2',
                'İşletim_Sistemi': 'Mac Os',
                'ram': '16',
                'ssd': '512'
            },
            'expected': (25000, 45000)
        }
    ]
    
    results = []
    for config in test_configs:
        price = test_configuration(
            config['name'], 
            config['data'], 
            config.get('expected')
        )
        if price:
            results.append((config['name'], price, config.get('expected')))
    
    # Summary
    print("\n" + "="*60)
    print("📈 SUMMARY")
    print("="*60)
    
    accurate_count = 0
    for name, price, expected in results:
        if expected and expected[0] <= price <= expected[1]:
            status = "✅ ACCURATE"
            accurate_count += 1
        elif expected:
            status = "⚠️  OUTSIDE RANGE"
        else:
            status = "✅ SUCCESS"
        
        print(f"{name}: {price:,.0f} TL {status}")
    
    if results:
        accuracy = (accurate_count / len(results)) * 100
        avg_price = sum(price for _, price, _ in results) / len(results)
        
        print(f"\n📊 Overall Accuracy: {accuracy:.1f}%")
        print(f"💰 Average Price: {avg_price:,.0f} TL")
        print(f"🔄 Successful Predictions: {len(results)}/{len(test_configs)}")

if __name__ == "__main__":
    main()
