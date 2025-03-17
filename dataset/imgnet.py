import os
import shutil

def aplanar_carpeta(destino):
    """Mueve todas las imágenes dentro de subcarpetas en 'destino' a 'destino' directamente."""
    
    # Recorrer todas las carpetas dentro de 'destino'
    for carpeta_raiz, subcarpetas, archivos in os.walk(destino, topdown=False):
        if carpeta_raiz == destino:
            continue  # Evitar procesar la carpeta raíz

        for archivo in archivos:
            archivo_origen = os.path.join(carpeta_raiz, archivo)
            archivo_destino = os.path.join(destino, archivo)

            # Evitar sobrescribir archivos con el mismo nombre
            if os.path.exists(archivo_destino):
                nombre, extension = os.path.splitext(archivo)
                contador = 1
                while os.path.exists(archivo_destino):
                    archivo_destino = os.path.join(destino, f"{nombre}_{contador}{extension}")
                    contador += 1

            # Mover archivo
            shutil.move(archivo_origen, archivo_destino)
            print(f"📦 Movido: {archivo_origen} → {archivo_destino}")

        # Si la carpeta está vacía después de mover los archivos, eliminarla
        if not os.listdir(carpeta_raiz):
            os.rmdir(carpeta_raiz)
            print(f"🗑️ Eliminada carpeta vacía: {carpeta_raiz}")

# 🔹 Uso del script
carpeta_destino = r"dataset/todas"
aplanar_carpeta(carpeta_destino)
 