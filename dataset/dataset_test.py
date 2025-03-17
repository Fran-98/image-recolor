import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def eliminar_imagenes_no_encontradas(directorio_origen, directorio_destino):
    """
    Elimina rápidamente las imágenes del directorio_origen que no existen en el directorio_destino.
    
    Args:
        directorio_origen: Ruta al directorio de origen donde se eliminarán imágenes
        directorio_destino: Ruta al directorio de referencia
    """
    # Tiempo de inicio
    tiempo_inicio = time.time()
    
    # Extensiones comunes de imágenes
    extensiones_imagenes = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    # Convertir las rutas a objetos Path
    origen = Path(directorio_origen)
    destino = Path(directorio_destino)
    
    # Verificar que ambos directorios existan
    if not origen.is_dir():
        print(f"Error: El directorio de origen '{directorio_origen}' no existe.")
        return
    
    if not destino.is_dir():
        print(f"Error: El directorio de destino '{directorio_destino}' no existe.")
        return
    
    print("Cargando archivos de los directorios...")
    
    # Crear un conjunto (set) con los nombres de archivos en destino para búsqueda rápida O(1)
    archivos_destino = set()
    for archivo in destino.iterdir():
        if archivo.is_file() and archivo.suffix.lower() in extensiones_imagenes:
            archivos_destino.add(archivo.name.lower())
    
    # Lista de archivos a eliminar
    archivos_a_eliminar = []
    
    # Identificar archivos a eliminar
    for archivo in origen.iterdir():
        if archivo.is_file() and archivo.suffix.lower() in extensiones_imagenes:
            if archivo.name.lower() not in archivos_destino:
                archivos_a_eliminar.append(archivo)
    
    total_archivos = len(archivos_a_eliminar)
    print(f"Se encontraron {total_archivos} archivos para eliminar.")
    
    # Función para eliminar un archivo
    def eliminar_archivo(archivo):
        try:
            archivo.unlink()
            return True
        except Exception as e:
            print(f"Error al eliminar {archivo.name}: {e}")
            return False
    
    # Eliminar archivos en paralelo usando múltiples hilos
    eliminados = 0
    if archivos_a_eliminar:
        print("Eliminando archivos...")
        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 2)) as executor:
            resultados = list(executor.map(eliminar_archivo, archivos_a_eliminar))
            eliminados = sum(resultados)
    
    tiempo_total = time.time() - tiempo_inicio
    print(f"\nProceso completado en {tiempo_total:.2f} segundos. Se eliminaron {eliminados} de {total_archivos} imágenes.")


if __name__ == "__main__":
    # Solicitar las rutas de los directorios al usuario
    dir_origen = "dataset\dataset_128\imgs_gray"
    dir_destino = "dataset\dataset_128\imgs_color"
    
    # Ejecutar la función principal
    eliminar_imagenes_no_encontradas(dir_origen, dir_destino)