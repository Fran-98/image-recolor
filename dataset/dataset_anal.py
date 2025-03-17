import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

def analizar_imagen(ruta_imagen):
    """
    Analiza una imagen individual y devuelve sus características.
    
    Args:
        ruta_imagen: Ruta completa a la imagen
        
    Returns:
        dict: Diccionario con las características de la imagen o None si hay error
    """
    try:
        # Intentar abrir la imagen con OpenCV
        img = cv2.imread(ruta_imagen)
        
        if img is None:
            # Si falla OpenCV, intentar con PIL
            img_pil = Image.open(ruta_imagen)
            ancho, alto = img_pil.size
            es_color = img_pil.mode != 'L'
            formato = os.path.splitext(ruta_imagen)[1].lower()
            tamanio = os.path.getsize(ruta_imagen) / 1024  # Tamaño en KB
            
            return {
                'ruta': ruta_imagen,
                'ancho': ancho,
                'alto': alto,
                'resolucion': f"{ancho}x{alto}",
                'relacion_aspecto': round(ancho / alto, 2) if alto != 0 else 0,
                'es_color': es_color,
                'formato': formato,
                'tamanio_kb': tamanio
            }
        
        # Si OpenCV funciona, extraer características
        alto, ancho = img.shape[:2]
        es_color = len(img.shape) == 3 and img.shape[2] == 3
        formato = os.path.splitext(ruta_imagen)[1].lower()
        tamanio = os.path.getsize(ruta_imagen) / 1024  # Tamaño en KB
        
        return {
            'ruta': ruta_imagen,
            'ancho': ancho,
            'alto': alto,
            'resolucion': f"{ancho}x{alto}",
            'relacion_aspecto': round(ancho / alto, 2) if alto != 0 else 0,
            'es_color': es_color,
            'formato': formato,
            'tamanio_kb': tamanio
        }
        
    except Exception as e:
        print(f"Error al analizar {ruta_imagen}: {e}")
        return None

def analizar_dataset(carpeta, max_workers=8):
    """
    Analiza un dataset completo de imágenes y genera estadísticas.
    
    Args:
        carpeta: Ruta a la carpeta que contiene las imágenes
        max_workers: Número máximo de hilos para procesamiento paralelo
    """
    print(f"Analizando dataset en: {carpeta}")
    start_time = time.time()
    
    # Recopilar todas las rutas de imágenes (incluyendo subcarpetas)
    rutas_imagenes = []
    extensiones_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    
    for root, _, archivos in os.walk(carpeta):
        for archivo in archivos:
            if os.path.splitext(archivo)[1].lower() in extensiones_validas:
                rutas_imagenes.append(os.path.join(root, archivo))
    
    total_imagenes = len(rutas_imagenes)
    print(f"Se encontraron {total_imagenes} imágenes para analizar.")
    
    # Procesar imágenes en paralelo
    resultados = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, resultado in enumerate(executor.map(analizar_imagen, rutas_imagenes)):
            if resultado:
                resultados.append(resultado)
            
            # Mostrar progreso
            if (i + 1) % 100 == 0 or (i + 1) == total_imagenes:
                print(f"Procesadas {i + 1}/{total_imagenes} imágenes ({((i + 1)/total_imagenes)*100:.1f}%)")
    
    # Convertir a DataFrame para análisis más fácil
    df = pd.DataFrame(resultados)
    
    # Calcular estadísticas básicas
    print("\n====== ESTADÍSTICAS DEL DATASET ======")
    print(f"Total de imágenes analizadas: {len(df)}")
    
    # Estadísticas por formato
    print("\n----- Formatos de imagen -----")
    formatos = df['formato'].value_counts()
    for formato, cantidad in formatos.items():
        print(f"{formato}: {cantidad} ({cantidad/len(df)*100:.1f}%)")
    
    # Estadísticas por resolución
    print("\n----- Resoluciones más comunes -----")
    resoluciones = df['resolucion'].value_counts()
    for i, (resolucion, cantidad) in enumerate(resoluciones.items()):
        if i < 10:  # Mostrar solo las 10 más comunes
            print(f"{resolucion}: {cantidad} ({cantidad/len(df)*100:.1f}%)")
    
    # Estadísticas de tamaño
    print("\n----- Estadísticas de tamaño -----")
    print(f"Tamaño promedio: {df['tamanio_kb'].mean():.2f} KB")
    print(f"Tamaño mínimo: {df['tamanio_kb'].min():.2f} KB")
    print(f"Tamaño máximo: {df['tamanio_kb'].max():.2f} KB")
    
    # Estadísticas de dimensiones
    print("\n----- Estadísticas de dimensiones -----")
    print(f"Ancho promedio: {df['ancho'].mean():.2f} píxeles")
    print(f"Ancho mínimo: {df['ancho'].min()} píxeles")
    print(f"Ancho máximo: {df['ancho'].max()} píxeles")
    print(f"Alto promedio: {df['alto'].mean():.2f} píxeles")
    print(f"Alto mínimo: {df['alto'].min()} píxeles")
    print(f"Alto máximo: {df['alto'].max()} píxeles")
    
    # Estadísticas de relación de aspecto
    print("\n----- Relaciones de aspecto -----")
    aspectos = df['relacion_aspecto'].value_counts().head(5)
    for aspecto, cantidad in aspectos.items():
        print(f"{aspecto:.2f}: {cantidad} ({cantidad/len(df)*100:.1f}%)")
    
    # Color vs. B/N
    print("\n----- Color vs. Blanco y Negro -----")
    color_count = df['es_color'].value_counts()
    print(f"Imágenes a color: {color_count.get(True, 0)} ({color_count.get(True, 0)/len(df)*100:.1f}%)")
    print(f"Imágenes en B/N: {color_count.get(False, 0)} ({color_count.get(False, 0)/len(df)*100:.1f}%)")
    
    # Tiempo total
    tiempo_total = time.time() - start_time
    print(f"\nTiempo total de análisis: {tiempo_total:.2f} segundos")
    
    # Guardar resultados detallados
    df.to_csv('analisis_dataset.csv', index=False)
    print("\nSe ha guardado el análisis detallado en 'analisis_dataset.csv'")
    
    return df

def visualizar_estadisticas(df):
    """
    Genera visualizaciones de las estadísticas del dataset.
    
    Args:
        df: DataFrame con los datos analizados
    """
    plt.figure(figsize=(18, 12))
    
    # 1. Distribución de resoluciones
    plt.subplot(2, 3, 1)
    top_resoluciones = df['resolucion'].value_counts().head(10)
    top_resoluciones.plot(kind='bar')
    plt.title('Top 10 Resoluciones')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 2. Distribución de formatos
    plt.subplot(2, 3, 2)
    df['formato'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Distribución de Formatos')
    plt.ylabel('')
    
    # 3. Histograma de tamaños
    plt.subplot(2, 3, 3)
    df['tamanio_kb'].plot(kind='hist', bins=50)
    plt.title('Distribución de Tamaños')
    plt.xlabel('Tamaño (KB)')
    
    # 4. Relación ancho vs alto (scatter plot)
    plt.subplot(2, 3, 4)
    plt.scatter(df['ancho'], df['alto'], alpha=0.5)
    plt.title('Ancho vs Alto')
    plt.xlabel('Ancho (píxeles)')
    plt.ylabel('Alto (píxeles)')
    
    # 5. Distribución de relación de aspecto
    plt.subplot(2, 3, 5)
    df['relacion_aspecto'].plot(kind='hist', bins=30)
    plt.title('Distribución de Relación de Aspecto')
    plt.xlabel('Relación de Aspecto')
    
    # 6. Color vs B/N
    plt.subplot(2, 3, 6)
    df['es_color'].value_counts().plot(kind='pie', labels=['Color', 'B/N'], autopct='%1.1f%%')
    plt.title('Color vs Blanco y Negro')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig('estadisticas_dataset.png', dpi=300, bbox_inches='tight')
    print("\nSe ha guardado la visualización en 'estadisticas_dataset.png'")
    
    # Mostrar la gráfica
    plt.show()

def analizar_dataset_principal(carpeta_dataset):
    """Función principal para analizar un dataset de imágenes."""
    df = analizar_dataset(carpeta_dataset)
    visualizar_estadisticas(df)
    return df

# Ejemplo de uso
if __name__ == "__main__":
    # Reemplaza con la ruta de tu dataset
    carpeta_dataset = "dataset/todas"
    analizar_dataset_principal(carpeta_dataset)