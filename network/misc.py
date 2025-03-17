import numpy as np
from skimage import color
import torch
import warnings
import torchvision.transforms as transforms

# Ignoramos unas warnings molestas de la conversion a cielab
warnings.filterwarnings(
    "ignore", 
    message="Conversion from CIE-LAB, via XYZ to sRGB color space resulted in"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lab_to_rgb_batch(L, ab):
    """
    Convierte un batch de tensores L y ab a RGB.
    Asume que L está en [0,1] escalado a [0,100], y ab en [-1,1] escalado a [-128,127].
    
    Parámetros:
        L: (B,1,H,W) -> Tensor de luminancia en [0,1]
        ab: (B,2,H,W) -> Tensor de crominancia en [-128,127]

    Retorna:
        rgb_batch: (B,3,H,W) -> Tensor de imagen en RGB en [0,1]
    """
    # print(f'Antes de escalar ab {ab.min()} | {ab.max()}')
    # Escalar correctamente
    L = L * 100  # De [0,1] a [0,100]
    # ab = (ab - 0.5) * 255  # De [0,1] a [-128,127]
    # print(f'ab {ab.min()} | {ab.max()}')
    # print(f'L {L.min()} | {L.max()}')
    # Clamping para evitar valores fuera de rango
    L = torch.clamp(L, 0, 100)
    ab = torch.clamp(ab, -127, 127)

    # Convertir de (B,1,H,W) y (B,2,H,W) a (B,H,W,1) y (B,H,W,2)
    L_np = L.detach().permute(0, 2, 3, 1).cpu().numpy()
    ab_np = ab.detach().permute(0, 2, 3, 1).cpu().numpy()
    # print(f'L_np shape {L_np.shape} | ab_np shape {ab_np.shape}')
    # Concatenar en (B, H, W, 3) y aplicar conversión vectorizada
    lab_np = np.concatenate([L_np, ab_np], axis=-1)
    rgb_np = color.lab2rgb(lab_np)  # Vectorizado en todo el batch

    # Convertir de vuelta a tensor de PyTorch en (B,3,H,W)
    rgb_batch = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()

    # Mantener el mismo dispositivo que L
    return rgb_batch.to(L.device)

class ConditionalResize:
    def __init__(self, size, interpolation=transforms.InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if img.size != (self.size, self.size):  # img.size is (width, height)
            return transforms.functional.resize(img, (self.size, self.size), self.interpolation)
        return img