import os
import csv
from PIL import Image

# Pfad zu dem Ordner mit den Bildern die umbenannt werden sollen
image_folder = '/Users/anouknormann/Desktop/PracticalDeepLearning/ImageProject/Objects_DataSet/train_data'
csv_file = '/Users/anouknormann/Desktop/PracticalDeepLearning/ImageProject/Objects_DataSet/train_data.csv'

# Annahme: Sie haben eine Methode, um die ClassId aus dem Dateinamen oder einer anderen Quelle zu bestimmen
def get_class_id(image_name):
    # Dummy Implementation
    # Hier könnten Sie Ihre eigene Logik implementieren, um die ClassId aus dem Bildnamen zu extrahieren
    return int(image_name.split('_')[1])  

# CSV Datei erstellen und Spalten beschiften
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Width', 'Height', 'ClassId', 'Path'])

    # Durch den Ordner mit den Bildern iterieren
    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Nur Bilddateien verarbeiten
            image_path = os.path.join(image_folder, image_name)
            with Image.open(image_path) as img:
                width, height = img.size

                # ClassId bestimmen
                class_id = get_class_id(image_name)

                # Informationen in die CSV-Datei schreiben
                writer.writerow([width, height, class_id, os.path.join(image_folder, image_name)])