#!/usr/bin/env python3
"""
Simple model performance test
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test konfigürasyonları
test_configs = [
    {
        'name': 'Budget Laptop',
        'brand': 'ACER',
        'ekran kartı': 'Intel UHD Graphics',
        'işlemci modeli': '1005G1',
        'ram': 8,
        'ssd': 256,
        'expected_range': (8000, 15000)
    },
    {
        'name': 'Gaming Laptop',
        'brand': 'ASUS',
        'ekran kartı': 'NVIDIA GeForce RTX 3050',
        'işlemci modeli': '12700H',
        'ram': 16,
        'ssd': 512,
        'expected_range': (20000, 35000)
    },
    {
        'name': 'High-end Laptop',
        'brand': 'MONSTER',
        'ekran kartı': 'NVIDIA GeForce RTX 4060',
        'işlemci modeli': '13700H',
        'ram': 32,
        'ssd': 1024,
        'expected_range': (40000, 70000)
    }
]

print("=== MODEL ACCURACY TEST ===")
print()

for i, config in enumerate(test_configs, 1):
    print(f"Test {i}: {config['name']}")
    print(f"Config: {config['brand']}, {config['işlemci modeli']}, {config['ekran kartı']}")
    print(f"RAM: {config['ram']}GB, SSD: {config['ssd']}GB")
    print(f"Expected: {config['expected_range'][0]:,} - {config['expected_range'][1]:,} TL")
    print("-" * 50)
