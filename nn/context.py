import torch
import numpy as np
from ultralytics import YOLO

# Cargar el modelo YOLOv11 pre-entrenado
model_yolo = YOLO("yolo11n.pt") # La version n es la mas chica, se puede ver los resultados con versiones mas grandes
model_yolo.eval()  # Modo evaluación
num_classes = len(model_yolo.names)

def get_yolo_context(img_gray: torch.Tensor):
    """
    Obtiene un vector condicional de la imagen en escala de grises utilizando YOLO.
    Se asume que img_gray es un tensor de forma (1, H, W).
    """
    # Convertir la imagen gris a 3 canales (replicando el canal)
    img_3ch = img_gray.repeat(3, 1, 1)  # forma (3, H, W)
    
    # Convertir a numpy y reordenar a formato HWC
    img_np = img_3ch.cpu().numpy().transpose(1, 2, 0)
    
    # Normalizar y convertir a uint8
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
    img_np = img_np.astype(np.uint8)

    # Realizar la detección con YOLO
    results = model_yolo(img_np)
    
    if not results:
        return torch.zeros(num_classes).unsqueeze(0)  # Vector de ceros si no hay resultados
    
    result = results[0]
    boxes = result.boxes
    
    if boxes is None or boxes.shape[0] == 0:
        return torch.zeros(num_classes).unsqueeze(0)  # Vector de ceros si no hay detecciones
    
    # Extraer clases detectadas
    classes = boxes.cls  # Obtener clases detectadas
    if isinstance(classes, np.ndarray):  # Necesito que sea un tensor de torch o se rompe
        classes = torch.from_numpy(classes)

    # Crear un histograma de las clases de yolo
    hist = torch.histc(classes, bins=num_classes, min=0, max=num_classes)
    
    # Normalizar el vector
    context_vec = hist / (hist.sum() + 1e-6)
    
    # Agregar dimensión de batch
    context_vec = context_vec.unsqueeze(0)  # (1, num_classes)
    
    return context_vec
