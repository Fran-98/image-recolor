import cv2
import os
import shutil
import numpy as np
from PIL import Image
import imagehash

def es_imagen_bn_preciso(image_path, umbral_color=3.0, debug=False):
    """
    Método simple y preciso para detectar imágenes en escala de grises.
    Analiza directamente la diferencia entre la imagen original y su versión convertida a escala de grises.
    
    Args:
        image_path: Ruta a la imagen
        umbral_color: Umbral de diferencia máxima permitida
        debug: Si es True, imprime información de diagnóstico
    
    Returns:
        Boolean: True si la imagen es escala de grises, False si es color
    """
    try:
        # Leer la imagen con OpenCV
        img = cv2.imread(image_path)
        
        if img is None:
            if debug:
                print(f"No se pudo leer la imagen: {image_path}")
            return False
        
        # Si la imagen tiene un solo canal, es definitivamente B/N
        if len(img.shape) == 2 or img.shape[2] == 1:
            if debug:
                print(f"Imagen con un solo canal (definitivamente B/N): {image_path}")
            return True
            
        # Convertir a escala de grises y luego de vuelta a BGR para comparar
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Calcular la diferencia absoluta entre la imagen original y la versión en escala de grises
        diff = cv2.absdiff(img, gray_bgr)
        
        # Calcular la diferencia media entre las dos imágenes
        mean_diff = np.mean(diff)
        
        if debug:
            print(f"Diferencia media entre original y escala de grises: {mean_diff:.2f} (Umbral: {umbral_color})")
            
        return mean_diff <= umbral_color
    
    except Exception as e:
        print(f"Error al procesar {image_path}: {e}")
        return False

def detectar_vinetas(image_path, margen=15, umbral_diferencia=80, debug=False):
    """
    Detecta imágenes con viñetas (completas o parciales en bordes).
    
    Args:
        image_path: Ruta a la imagen
        margen: Tamaño del borde a examinar
        umbral_diferencia: Umbral para considerar una diferencia significativa
        debug: Si es True, imprime información de diagnóstico
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    alto, ancho = img.shape

    # Examinar múltiples regiones del borde
    borde_sup = img[:margen, :].mean()
    borde_inf = img[-margen:, :].mean()
    borde_izq = img[:, :margen].mean()
    borde_der = img[:, -margen:].mean()
    
    # Usar el centro de la imagen como referencia
    centro = img[alto//4:3*alto//4, ancho//4:3*ancho//4].mean()

    # Calcular diferencias con el centro
    dif_sup = abs(borde_sup - centro)
    dif_inf = abs(borde_inf - centro)
    dif_izq = abs(borde_izq - centro)
    dif_der = abs(borde_der - centro)
    
    if debug:
        print(f"Viñetas - {image_path}:")
        print(f"  Diferencias: sup={dif_sup:.1f}, inf={dif_inf:.1f}, izq={dif_izq:.1f}, der={dif_der:.1f}")
        print(f"  Umbral: {umbral_diferencia}")

    # Detectar viñetas con criterios más flexibles
    tiene_vineta_completa = (dif_sup > umbral_diferencia and dif_inf > umbral_diferencia and 
                            dif_izq > umbral_diferencia and dif_der > umbral_diferencia)
    tiene_vineta_parcial = ((dif_sup > umbral_diferencia and dif_inf > umbral_diferencia) or
                           (dif_izq > umbral_diferencia and dif_der > umbral_diferencia))
    
    if tiene_vineta_completa:
        return "completa"
    elif tiene_vineta_parcial:
        return "parcial"
    
    return None

def calcular_hash(image_path):
    """
    Calcula el hash perceptual de una imagen.
    
    Args:
        image_path: Ruta a la imagen
    
    Returns:
        imagehash: Hash perceptual de la imagen o None si hay error
    """
    try:
        img = Image.open(image_path).convert("L")  # Convertir a escala de grises
        return imagehash.phash(img)  # Hash perceptual
    except Exception as e:
        print(f"❌ Error procesando {image_path}: {e}")
        return None

def limpiar_dataset(carpeta_origen, carpeta_destino, modo_debug=False, limite_archivos=None):
    """
    Mueve imágenes en blanco y negro, con viñetas o duplicadas a la carpeta de descarte.
    
    Args:
        carpeta_origen: Carpeta donde están las imágenes originales
        carpeta_destino: Carpeta donde se moverán las imágenes descartadas
        modo_debug: Si es True, imprime información detallada
        limite_archivos: Número máximo de archivos a procesar (para pruebas)
    """
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    hashes = {}  # Diccionario para almacenar los hashes de imágenes únicas
    
    # Estadísticas
    total = 0
    descartadas_bn = 0
    descartadas_vineta = 0
    descartadas_duplicadas = 0
    
    archivos = os.listdir(carpeta_origen)
    if limite_archivos:
        archivos = archivos[:limite_archivos]

    for archivo in archivos:
        ruta_completa = os.path.join(carpeta_origen, archivo)
        
        if not os.path.isfile(ruta_completa):
            continue  # Saltar si no es un archivo
            
        total += 1
        if total % 10 == 0:
            print(f"Procesando archivo {total}/{len(archivos)}")

        motivo_descarte = None  # Para almacenar el motivo de descarte
        
        # Verificar si la imagen es blanco y negro con método preciso
        if es_imagen_bn_preciso(ruta_completa, umbral_color=2.0, debug=modo_debug):
            motivo_descarte = "blanco y negro"
            descartadas_bn += 1

        # Verificar si tiene viñetas
        if not motivo_descarte:
            tipo_vineta = detectar_vinetas(ruta_completa, debug=modo_debug)
            if tipo_vineta and False:
                motivo_descarte = f"viñeta {tipo_vineta}"
                descartadas_vineta += 1

        # Verificar duplicados
        if not motivo_descarte:
            img_hash = calcular_hash(ruta_completa)
            if img_hash:
                if img_hash in hashes:
                    motivo_descarte = f"duplicada de {hashes[img_hash]}"
                    descartadas_duplicadas += 1
                else:
                    hashes[img_hash] = archivo  # Guardar la primera imagen encontrada

        # Mover la imagen si cumple alguna condición de descarte
        if motivo_descarte:
            shutil.move(ruta_completa, os.path.join(carpeta_destino, archivo))
            print(f"🚮 Movida a descarte: {archivo} - Motivo: {motivo_descarte}")
    
    # Mostrar estadísticas
    print("\n==== Estadísticas ====")
    print(f"Total de archivos procesados: {total}")
    print(f"Imágenes descartadas por B/N: {descartadas_bn} ({descartadas_bn/total*100:.1f}%)")
    print(f"Imágenes descartadas por viñetas: {descartadas_vineta} ({descartadas_vineta/total*100:.1f}%)")
    print(f"Imágenes descartadas por duplicados: {descartadas_duplicadas} ({descartadas_duplicadas/total*100:.1f}%)")
    print(f"Total de imágenes descartadas: {descartadas_bn + descartadas_vineta + descartadas_duplicadas} ({(descartadas_bn + descartadas_vineta + descartadas_duplicadas)/total*100:.1f}%)")

# Función adicional para hacer pruebas individuales
def probar_imagen(ruta_imagen):
    """
    Prueba individualmente una imagen para ver si es detectada como B/N.
    Útil para ajustar el umbral correcto.
    """
    print(f"Analizando imagen: {ruta_imagen}")
    
    # Probamos con diferentes umbrales
    umbrales = [1.0, 2.0, 3.0, 5.0, 10.0]
    for umbral in umbrales:
        resultado = es_imagen_bn_preciso(ruta_imagen, umbral_color=umbral, debug=True)
        print(f"Umbral {umbral}: {'B/N' if resultado else 'COLOR'}")
    
    print("\nVisualización (valores más altos indican más color):")
    # Visualizar la diferencia entre la imagen original y la versión en escala de grises
    img = cv2.imread(ruta_imagen)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        diff = cv2.absdiff(img, gray_bgr)
        print(f"Diferencia media: {np.mean(diff):.2f}")
        
        # Opcional: guardar imagen diferencia para visualización
        # cv2.imwrite("diferencia.jpg", diff * 20)  # Amplificar diferencias para ver mejor

# probar_imagen(r'dataset/todas/ILSVRC2012_val_00000622.JPEG')
# 🔹 Uso del script
carpeta_origen = r"dataset/todas"
carpeta_destino = r"dataset/descarte"
limpiar_dataset(carpeta_origen, carpeta_destino)
