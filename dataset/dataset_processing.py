import os
import argparse
from PIL import Image
from vintager import convert

def process_image(image_path, out_gray_path, out_color_path):
    """
    Procesa una imagen: genera y guarda la versión en color y la versión en blanco y negro.
    
    Parámetros:
      - image_path: ruta de la imagen original.
      - out_gray_path: ruta de salida para la imagen en escala de grises.
      - out_color_path: ruta de salida para la imagen a color.
    """
    # Abrir la imagen original (color) con PIL
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error abriendo {image_path}: {e}")
        return
    
    # Guardar la imagen a color en el directorio de salida
    im.save(out_color_path)
    
    # Generar la imagen en blanco y negro usando vintager librería creada para este proyecto.
    # Aquí configuramos:
    #   black_and_white=True: queremos la versión en escala de grises con alto contraste, es el mejor resultado comparado a solo blanco y negro.
    im_gray = convert(
        image_path,
        None,  # No se guarda automáticamente, lo haremos con PIL.
        black_and_white=True,
        high_contrast_black_and_white=False,
        apply_sepia=False,
        apply_vintage=0,
        apply_grain=False,
        noise_level=0
    )
    
    # Guardar la imagen en escala de grises en el directorio de salida
    im_gray.save(out_gray_path)
    print(f"Procesada: {os.path.basename(image_path)}")

def main(input_folder, output_folder_gray, output_folder_color):
    """
    Recorre el directorio de entrada y procesa cada imagen.
    Crea dos carpetas de salida (si no existen): una para imágenes en escala de grises y otra para las imágenes a color.
    """
    # Crear las carpetas de salida si no existen
    os.makedirs(output_folder_gray, exist_ok=True)
    os.makedirs(output_folder_color, exist_ok=True)
    
    # Recorrer todos los archivos del directorio de entrada
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            input_path = os.path.join(input_folder, filename)
            out_gray_path = os.path.join(output_folder_gray, filename)
            out_color_path = os.path.join(output_folder_color, filename)
            
            process_image(input_path, out_gray_path, out_color_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para formatear un dataset de imágenes para entrenamiento.\n"
                                                 "Genera pares de imagen a color y su correspondiente imagen en blanco y negro.")
    parser.add_argument("--input_folder", type=str, required=True, 
                        help="Ruta del directorio que contiene las imágenes a color originales.")
    parser.add_argument("--output_folder_gray", type=str, required=True, 
                        help="Ruta del directorio donde se guardarán las imágenes en blanco y negro.")
    parser.add_argument("--output_folder_color", type=str, required=True, 
                        help="Ruta del directorio donde se guardarán las imágenes a color.")
    
    args = parser.parse_args()
    main(args.input_folder, args.output_folder_gray, args.output_folder_color)
