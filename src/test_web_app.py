#!/usr/bin/env python3
"""
Test script for the web application prediction endpoint
"""
import requests
import json
from pprint import pprint

# Test configurations
test_configs = [
    {
        "name": "Standard Gaming Laptop",
        "data": {
            'brand': 'MSI',
            'ekran kartı': 'NVIDIA GeForce RTX 3060',
            'işlemci modeli': 'Intel Core i7-12700H',
            'ekran kartı hafızası': '6',
            'temel işlemci hızı': '2.3',
            'maksimum işlemci hızı': '4.7',
            'ram': '16',
            'ssd': '512',
            'İşletim_Sistemi': 'Windows'
        }
    },
    {
        "name": "Budget Laptop with Fuzzy GPU",
        "data": {
            'brand': 'LENOVO',
            'ekran kartı': 'RTX 3050',  # Test fuzzy matching
            'işlemci modeli': '12500H',
            'ekran kartı hafızası': '4',
            'temel işlemci hızı': '2.5',
            'maksimum işlemci hızı': '4.2',
            'ram': '8',
            'ssd': '256',
            'İşletim_Sistemi': 'Windows'
        }
    },
    {
        "name": "Apple MacBook Test",
        "data": {
            'brand': 'Apple',
            'ekran kartı': 'Apple M2',
            'işlemci modeli': 'Apple M2',
            'ekran kartı hafızası': '',
            'temel işlemci hızı': '',
            'maksimum işlemci hızı': '',
            'ram': '16',
            'ssd': '512',
            'İşletim_Sistemi': 'Mac Os'
        }
    }
]

def test_prediction(config):
    """Test a single prediction"""
    print(f"\n🧪 Testing: {config['name']}")
    print("=" * 50)
    
    try:
        response = requests.post('http://127.0.0.1:5003/predict', data=config['data'])
        
        if response.status_code == 200:
            result = response.json()
            
            if result['success']:
                print(f"✅ Prediction successful!")
                print(f"💰 Price: {result['prediction']:,.2f} TL")
                
                if result.get('status') != 'normal':
                    print(f"⚠️  Status: {result['status']}")
                    
                if result.get('warning'):
                    print(f"⚠️  Warning: {result['warning']}")
                
                if result.get('data_issues'):
                    print(f"\n🔄 Data mappings:")
                    for issue in result['data_issues']:
                        print(f"   {issue['field']}: '{issue['original']}' → '{issue['mapped_to']}'")
                
                if result.get('recommendations'):
                    print(f"\n💡 Recommendations:")
                    for rec in result['recommendations']:
                        print(f"   • {rec}")
                        
            else:
                print(f"❌ Prediction failed: {result['error']}")
                
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("🚀 Testing Web Application Prediction Endpoint")
    print("=" * 60)
    
    # Test if server is running
    try:
        response = requests.get('http://127.0.0.1:5003/')
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server returned error:", response.status_code)
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please make sure the Flask app is running on port 5003")
        return
    
    # Run tests
    for config in test_configs:
        test_prediction(config)
    
    print(f"\n🏁 Testing complete!")

if __name__ == "__main__":
    main()
