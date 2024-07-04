import os
from rembg import remove, new_session
import cv2
import numpy as np

# Directorios de entrada y salida
input_folder = 'imagenes'
output_folder = 'SalidaCaly'

# Crear una sesión nueva con el modelo isnet-anime
session = new_session("isnet-anime")

# Función para crear la estructura de carpetas
def create_directory_structure(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        # Construir la estructura de carpetas en el directorio de salida
        for file in files:
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_dir, os.path.relpath(root, input_dir), file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

def process_images(input_folder, output_folder):
    # Crear la estructura de carpetas en la carpeta de salida
    create_directory_structure(input_folder, output_folder)

    # Recorrer todas las imágenes en las subcarpetas de "Caly"
    for root, _, files in os.walk(input_folder):
        for file in files:
            # Ruta de la imagen de entrada
            input_path = os.path.join(root, file)
            # Ruta de la imagen de salida
            output_path = os.path.join(output_folder, os.path.relpath(root, input_folder), file)

            # Cargar la imagen usando OpenCV
            input_image = cv2.imread(input_path)
            
            # Remover el fondo
            output_image = remove(input_image, session=session)

            # Convertir la imagen de salida a formato BGRA
            output_image_bgra = cv2.cvtColor(output_image, cv2.COLOR_RGBA2BGRA)

            # Crear una máscara booleana para los píxeles opacos
            mask = output_image_bgra[:, :, 3] > 2

            # Aplicar la máscara a la imagen original
            input_image_masked = np.zeros_like(input_image)
            input_image_masked[mask] = input_image[mask]
            output_image_masked = remove(input_image_masked, session=session)
            # Guardar la imagen resultante recortada
            cv2.imwrite(output_path, input_image_masked)

            print(f"Procesada: {input_path} -> {output_path}")

# Procesar todas las imágenes en las subcarpetas de "Caly"
process_images(input_folder, output_folder)
