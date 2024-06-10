import os

# Pfad zu dem Ordner mit den Bildern
image_folder = '/Users/anouknormann/Desktop/PracticalDeepLearning/ImageProject/Objects_DataSet/flipchart_reduziert'

# Unterstützte Bildformate
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

# Klasse des Objekts im Bild -> 1:stuhl, 2:tisch, 3:flipchart
class_id = 0

# Bilddateien im Ordner auflisten
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]

# Bilder umbenennen
for idx, image_name in enumerate(image_files, start=1):
    # Neues Bildname erstellen
    new_image_name =  f'img_{class_id}_{idx}_{os.path.splitext(image_name)[1]}'
    old_image_path = os.path.join(image_folder, image_name)
    new_image_path = os.path.join(image_folder, new_image_name)
    
    # Bild umbenennen
    os.rename(old_image_path, new_image_path)
    print(f'Renamed: {image_name} -> {new_image_name}')

print('Renaming completed.')
