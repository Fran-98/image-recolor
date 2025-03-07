import numpy as np
from skimage import color
import torch
import warnings

# Ignoramos unas warnings molestas de la conversion a cielab
warnings.filterwarnings(
    "ignore", 
    message="Conversion from CIE-LAB, via XYZ to sRGB color space resulted in"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lab_to_rgb_batch(L, ab):
    """
    Convierte un batch de tensores L y ab a RGB.
    Asume que L está en [0,1] escalado a [0,100], y ab en [0,1] escalado a [-128,127].
    L: (B,1,H,W)
    ab: (B,2,H,W)
    Retorna un tensor (B,3,H,W) en [0,1].
    """
    B = L.size(0)
    # print(f"No escalado ab range: {ab.min().item()} | {ab.max().item()}")
    # Escalar correctamente antes de clamping
    L = L * 100
    # ab = ab * 255 - 127
    ab = (ab - 0.5) * 255
    # ab = ab * 127
    
    
    # Aplicar clamp poor las dudas
    L = torch.clamp(L, 0, 100)
    ab = torch.clamp(ab, -127, 127)

    # print(f"Escalado ab range: {ab.min().item()} | {ab.max().item()}")
    # Convertir a NumPy y pasamos a cpu mientras
    L_np = L.detach().cpu().numpy()
    ab_np = ab.detach().cpu().numpy()
    # print(f"NumPy ab range after conversion: {ab_np.min()} | {ab_np.max()}")

    # print(f"L range: {L.min().item()} | {L.max().item()}")
    # print(f"ab range: {ab.min().item()} | {ab.max().item()}")

    rgb_list = []
    
    for i in range(B):
        # Extraer L, a, b para la i-ésima imagen
        L_i = L_np[i, 0]
        a_i = ab_np[i, 0]
        b_i = ab_np[i, 1]

        lab_img = np.stack([L_i, a_i, b_i], axis=-1).astype(np.float32) # Nos aseguramos de que sea float32 el tipo
        lab_img = np.clip(lab_img, [0, -128, -128], [100, 127, 127])

        rgb_img = color.lab2rgb(lab_img)  # (H,W,3) en [0,1]

        # Convertir a tensor PyTorch en formato (3, H, W)
        rgb_tensor = torch.from_numpy(rgb_img.transpose((2, 0, 1))).float()
        rgb_list.append(rgb_tensor.unsqueeze(0))  # (1,3,H,W)
    
    # Concatenar a lo largo del batch
    rgb_batch = torch.cat(rgb_list, dim=0)  # (B,3,H,W)
    return rgb_batch.to(device)  # Regresarlo al mismo device
