import os

# Dizin yolları
directories = [
    './Dataset-Places2/train',
    './Dataset-Places2/val',
    './Dataset-Places2/test/',
    './Saved-models/Saved-gen/',
    './Saved-models/Saved-disc/',
    './Saved-models/Saved-weights/',
    './Plots/',
    './Saved-models/Saved-gen-pre/',
    './Saved-models/Saved-disc-pre/'
]

# Dizinleri kontrol et ve yoksa oluştur
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"{directory} oluşturuldu.")
    else:
        print(f"{directory} zaten mevcut.")
