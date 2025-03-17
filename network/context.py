import torch
import torch.nn.functional as F
from ultralytics import YOLO
import cv2
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_yolo = YOLO("yolo11n.pt").to(device)
model_yolo.eval()
num_classes = 256

def extraer_bordes(img_l):
    """Aplica Canny para extraer bordes del canal L."""
    img_l_uint8 = (img_l * 255).astype(np.uint8)  # Convertir a 8 bits
    bordes = cv2.Canny(img_l_uint8, 100, 200)  # Detectar bordes con Canny
    bordes = bordes / 255.0  # Normalizar a [0,1]
    return bordes

def procesar_entrada(img_path):
    """Carga la imagen, la convierte a Lab y extrae bordes."""
    img = cv2.imread(img_path)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32) / 255.0
    img_l = img_lab[:, :, 0]  # Extraer canal L
    bordes = extraer_bordes(img_l)  # Detectar bordes

    entrada = np.stack([img_l, bordes], axis=-1)  # Concatenar L y bordes
    return entrada

def get_yolo_context(img_gray: torch.Tensor):
    """
    Extraemos las primeras capas de YOLOv11 y pasamos nuestra imagen en blanco y negro para tener un vector de contexto.
    """

    img_gray = img_gray[:, 0:1, :, :] # Esto es para separar solo el canal L ya que ahora tenemos bordes tambien

    img_gray = img_gray.to(device)

    img_3ch = img_gray.repeat(1, 3, 1, 1) # Necesito 3 canales para yolo
    
    backbone = model_yolo.model.model[:9] # EN YOLOv11 el backbone es hasta la capa 8 (slice hasta 9 porque excluye esta)

    with torch.no_grad():
        features = backbone(img_3ch)  # (B, C, H', W')
    
    pooled = F.adaptive_avg_pool2d(features, (1, 1))  # shape: (B, C, 1, 1)
    context_vector = pooled.view(pooled.size(0), -1)  # shape: (B, C)
    # print(f'Context len {context_vector.size()}')
    return context_vector

# Estructura de yolo, hasta donde?
# <class 'ultralytics.nn.modules.conv.Conv'>                                                                                                                                    | 0/10 [00:00<?, ?it/s] 
# <class 'ultralytics.nn.modules.conv.Conv'>
# <class 'ultralytics.nn.modules.block.C3k2'>
# <class 'ultralytics.nn.modules.conv.Conv'>
# <class 'ultralytics.nn.modules.block.C3k2'>
# <class 'ultralytics.nn.modules.conv.Conv'>
# <class 'ultralytics.nn.modules.block.C3k2'>
# <class 'ultralytics.nn.modules.conv.Conv'>
# <class 'ultralytics.nn.modules.block.C3k2'> <--- Hasta acá el backbone
# <class 'ultralytics.nn.modules.block.SPPF'>
# <class 'ultralytics.nn.modules.block.C2PSA'>
# <class 'torch.nn.modules.upsampling.Upsample'>
# <class 'ultralytics.nn.modules.conv.Concat'>
# <class 'ultralytics.nn.modules.block.C3k2'>
# <class 'torch.nn.modules.upsampling.Upsample'>
# <class 'ultralytics.nn.modules.conv.Concat'>
# <class 'ultralytics.nn.modules.block.C3k2'>
# <class 'ultralytics.nn.modules.conv.Conv'>
# <class 'ultralytics.nn.modules.conv.Concat'>
# <class 'ultralytics.nn.modules.block.C3k2'>
# <class 'ultralytics.nn.modules.conv.Conv'>
# <class 'ultralytics.nn.modules.conv.Concat'>
# <class 'ultralytics.nn.modules.block.C3k2'> <--- Hasta acá el Neck
# <class 'ultralytics.nn.modules.head.Detect'> <--- Obviamente esta ya es la salida