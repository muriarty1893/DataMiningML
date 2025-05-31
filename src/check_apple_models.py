#!/usr/bin/env python3
"""
Check what Apple models are available in the label encoders
"""
import joblib
import os

# Load label encoders
label_encoders_path = os.path.join('..', 'models', 'label_encoders.joblib')
label_encoders = joblib.load(label_encoders_path)

print("=== APPLE CPU MODELS IN TRAINING DATA ===")
if 'İşlemci_Modeli' in label_encoders:
    cpu_classes = label_encoders['İşlemci_Modeli'].classes_
    apple_cpus = [cpu for cpu in cpu_classes if 'apple' in str(cpu).lower() or 'm1' in str(cpu).lower() or 'm2' in str(cpu).lower()]
    print(f"Found {len(apple_cpus)} Apple CPU models:")
    for cpu in sorted(apple_cpus):
        print(f"  - {cpu}")
else:
    print("İşlemci_Modeli not found in label encoders")

print("\n=== APPLE GPU MODELS IN TRAINING DATA ===")
if 'Ekran_Kartı' in label_encoders:
    gpu_classes = label_encoders['Ekran_Kartı'].classes_
    apple_gpus = [gpu for gpu in gpu_classes if 'apple' in str(gpu).lower() or 'm1' in str(gpu).lower() or 'm2' in str(gpu).lower()]
    print(f"Found {len(apple_gpus)} Apple GPU models:")
    for gpu in sorted(apple_gpus):
        print(f"  - {gpu}")
else:
    print("Ekran_Kartı not found in label encoders")

print("\n=== ALL AVAILABLE CPU MODELS (first 20) ===")
if 'İşlemci_Modeli' in label_encoders:
    cpu_classes = sorted(label_encoders['İşlemci_Modeli'].classes_)
    for cpu in cpu_classes[:20]:
        print(f"  - {cpu}")
    print(f"  ... and {len(cpu_classes)-20} more")

print("\n=== ALL AVAILABLE GPU MODELS (first 20) ===")
if 'Ekran_Kartı' in label_encoders:
    gpu_classes = sorted(label_encoders['Ekran_Kartı'].classes_)
    for gpu in gpu_classes[:20]:
        print(f"  - {gpu}")
    print(f"  ... and {len(gpu_classes)-20} more")
