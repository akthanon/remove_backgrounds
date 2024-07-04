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

def increase_contrast(image, intensity=1.0):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Ajustar el límite de clip según la intensidad deseada
    clip_limit = 3.0 * intensity
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def decrease_contrast(original_image, processed_image):
    # Convertir la imagen original a RGBA si es necesario
    if original_image.shape[2] == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
    
    # Crear una máscara para el fondo eliminado
    gray_processed = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray_processed == 0, 0, 255).astype('uint8')
    
    # Copiar la imagen original y aplicar la máscara
    output_image = original_image.copy()
    output_image[mask == 0] = processed_image[mask == 0]

    return output_image

def process_images(input_folder, output_folder, contrast_intensity=1.0):
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

            # Aumentar el contraste de la imagen con la intensidad especificada
            high_contrast_image = increase_contrast(input_image, intensity=contrast_intensity)
            
            # Remover el fondo
            no_bg_image = remove(high_contrast_image, session=session)

            # Restaurar el contraste original de la imagen sin fondo
            output_image = decrease_contrast(input_image, no_bg_image)

            # Guardar la imagen resultante
            cv2.imwrite(output_path, output_image)

            print(f"Procesada: {input_path} -> {output_path}")

# Procesar todas las imágenes en las subcarpetas de "Caly" con un contraste moderado (intensidad = 0.5)
process_images(input_folder, output_folder, contrast_intensity=1)
