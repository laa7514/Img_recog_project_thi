from PIL import Image
import os

# Pfad zum Ordner mit den Bildern
folder_path = '/Users/anouknormann/Desktop/PracticalDeepLearning/ImageProject/Objects_DataSet/flipchart'

# Zielordner für die komprimierten Bilder
output_folder = 'ImageProject/Objects_DataSet/flipchart_reduziert'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Faktor zur Reduzierung der Bildqualität (0-100, 100 = beste Qualität)
quality = 2

# Durch alle Dateien im Ordner iterieren
for file_name in os.listdir(folder_path):
    # Sicherstellen, dass es sich um eine Bilddatei handelt
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Bild öffnen
        image_path = os.path.join(folder_path, file_name)
        with Image.open(image_path) as img:
            # Neuen Dateinamen erstellen
            new_file_name = file_name
            # Zielbildpfad
            output_path = os.path.join(output_folder, new_file_name)
            # Bild speichern mit reduzierter Qualität
            img.save(output_path, optimize=True, quality=quality)
            print(f'Compressed image saved: {new_file_name}')

print('Compression completed.')
