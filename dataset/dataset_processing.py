import os
from PIL import Image
from vintager import convert
from tqdm import tqdm

def process_image(image_path, out_gray_path, out_color_path, size=(128, 128)):
    """
    Procesa una imagen: genera y guarda la versión en color y la versión en blanco y negro,
    ambas redimensionadas a un tamaño específico.
    
    Parámetros:
      - image_path: ruta de la imagen original.
      - out_gray_path: ruta de salida para la imagen en escala de grises.
      - out_color_path: ruta de salida para la imagen a color.
      - size: tupla con las dimensiones de destino para el redimensionado (ancho, alto).
    """
    # Abrir la imagen original (color) con PIL
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error abriendo {image_path}: {e}")
        return
    
    # Redimensionar la imagen a las dimensiones deseadas (256x256)
    im = im.resize(size)
    
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
    
    # Redimensionar la imagen en blanco y negro a las dimensiones deseadas (256x256)
    im_gray = im_gray.resize(size)
    
    # Guardar la imagen en escala de grises en el directorio de salida
    im_gray.save(out_gray_path)
    # print(f"Procesada: {os.path.basename(image_path)}")

def main(input_folder, output_folder_gray, output_folder_color, size=(128, 128)):
    """
    Recorre el directorio de entrada y procesa cada imagen.
    Crea dos carpetas de salida (si no existen): una para imágenes en escala de grises y otra para las imágenes a color.
    """
    # Crear las carpetas de salida si no existen
    os.makedirs(output_folder_gray, exist_ok=True)
    os.makedirs(output_folder_color, exist_ok=True)
    
    # Recorrer todos los archivos del directorio de entrada
    for filename in tqdm(os.listdir(input_folder), desc="Procesando imágenes"):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            input_path = os.path.join(input_folder, filename)
            out_gray_path = os.path.join(output_folder_gray, filename)
            out_color_path = os.path.join(output_folder_color, filename)
            
            process_image(input_path, out_gray_path, out_color_path, size)

if __name__ == '__main__':
    input_folder = 'dataset/todas_paisajes'
    output_folder_gray = 'dataset/dataset_128_paisajes/imgs_gray'
    output_folder_color = 'dataset/dataset_128_paisajes/imgs_color'
    main(input_folder, output_folder_gray, output_folder_color, size=(128, 128))

