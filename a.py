import cv2
import os

# Kaynak ve hedef dizinler
SOURCE_DIR = "./Dataset-Places2/val/color"  # Kaynak resimler burada
DEST_DIR = "./Dataset-Places2/val/grayscale"     # Grayscale resimler burada saklanacak

# Kaynak dizindeki tüm resimleri al
def convert_images_to_grayscale(source_dir, dest_dir):
    # Eğer hedef dizin yoksa oluştur
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Klasördeki tüm resimleri gez
    for filename in os.listdir(source_dir):
        # Dosyanın tam yolunu al
        img_path = os.path.join(source_dir, filename)
        
        # Resmi oku (OpenCV ile)
        img = cv2.imread(img_path)
        
        # Eğer resim var ise, siyah-beyaz'a çevir
        if img is not None:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Yeni dosya adını oluştur
            dest_path = os.path.join(dest_dir, filename)
            
            # Siyah-beyaz resmi kaydet
            cv2.imwrite(dest_path, gray_img)
            print(f"Grayscale resim kaydedildi: {dest_path}")
        else:
            print(f"Resim okunamadı: {img_path}")
    print("Tüm resimler kaydedildi")        

# Kullanım
convert_images_to_grayscale(SOURCE_DIR, DEST_DIR)
