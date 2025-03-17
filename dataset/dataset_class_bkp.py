from PIL import Image
from torch.utils.data import Dataset
import os

import torch
import numpy as np
from skimage import color  

import matplotlib.pyplot as plt
from network.colorfulness import get_gamut_colors_custom

gamut_colors = get_gamut_colors_custom(grid_size=10, L_value=50, tolerance=0.1)
ab_centroids  = np.array([[c[3], c[4]] for c in gamut_colors])
ab_centroids  = torch.tensor(ab_centroids , dtype=torch.float32).to('cuda')


def soft_encode(img_ab, ab_centroids, sigma=5, show_heatmap = False):
    """
    Realiza el soft-encoding de los valores ab a un vector Z de probabilidades.

    Args:
        img_ab: Tensor de valores ab [B, 2, H, W].
        ab_centroids: Tensor de centroides de los bins [Q, 2].
        sigma: Desviación estándar para la ponderación Gaussiana.

    Returns:
        Tensor Z [B, H, W, Q] con la distribución de probabilidad soft-encoded.
    """
    img_ab = img_ab.unsqueeze(1).to('cuda')  # [B, 1, 2, H, W]
    img_ab = (img_ab * 254) - 127

    B, _, H, W = img_ab.shape
    Q = ab_centroids.shape[0]



    # Expandir dimensiones para broadcasting
    img_ab = img_ab = img_ab.permute(0, 2, 3, 1).unsqueeze(3)  # [B, H, W, 1, 2]
    ab_centroids = ab_centroids.view(1, 1, 1, Q, 2)  # [1, 1, 1, Q, 2]

    # Calcular distancias cuadradas
    distances_sq = torch.sum((img_ab - ab_centroids) ** 2, dim=4)  # [B, H, W, Q]

    # Encontrar los 5 vecinos más cercanos
    _, top5_indices = torch.topk(distances_sq, k=5, dim=3, largest=False)  # [B, H, W, 5]

    # Crear tensor Z de ceros
    Z = torch.zeros((B, H, W, Q), device=img_ab.device)

    # Obtener distancias de los 5 vecinos
    distances = torch.gather(distances_sq, dim=3, index=top5_indices)  # [B, H, W, 5]
    distances = torch.sqrt(distances)

    # Calcular pesos Gaussianos
    weights = torch.exp(-distances**2 / (2 * sigma**2))
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)  # Normalizar

    # Indexar y asignar valores de manera eficiente
    BHW = B * H * W
    offsets = torch.arange(BHW, device=img_ab.device).view(B, H, W) * Q
    indices_flat = (top5_indices + offsets.unsqueeze(-1)).view(-1)
    weights_flat = weights.view(-1)

    # Aplicar index_add_ en lugar de bucle
    Z.view(-1).index_add_(0, indices_flat, weights_flat)

    # Normalizar sobre la dimensión Q para asegurar que sea una distribución de probabilidad
    Z = Z / Z.sum(dim=-1, keepdim=True)
    if show_heatmap:
        # Visualizar el primer mapa de calor del batch
        heatmap = Z[0].sum(dim=2).detach().cpu().numpy()  # Sumar sobre la dimensión Q para visualizar
        plt.imshow(heatmap)
        plt.title("Soft Encode Heatmap")
        plt.colorbar()
        plt.show()
    return Z

def soft_decode(Z, ab_centroids, T=0.38, show_heatmap=False):
    """
    Convierte la distribución de probabilidad Z en valores ab usando el annealed-mean.

    Args:
        Z: Tensor de distribución de probabilidad [B, H, W, Q].
        ab_centroids: Tensor de centroides de los bins [Q, 2].
        T: Temperatura para el ajuste de la distribución.

    Returns:
        Tensor de valores ab [B, 2, H, W].
    """
    # Ajustar la temperatura
    Z_T = torch.exp(torch.log(Z + 1e-8) / T)  # Evitar log(0) con 1e-8
    Z_T = Z_T / Z_T.sum(dim=-1, keepdim=True)  # Renormalizar

    # Obtener el valor esperado (annealed mean)
    ab_pred = torch.sum(Z_T.unsqueeze(-1) * ab_centroids.view(1, 1, 1, -1, 2), dim=3)

    if show_heatmap:
        # Visualizar el primer mapa de calor del batch
        heatmap = Z_T[0].sum(dim=2).detach().cpu().numpy()  # Sumar sobre la dimensión Q para visualizar
        plt.imshow(heatmap)
        plt.title("Soft Decode Heatmap")
        plt.colorbar()
        plt.show()
    
    return ab_pred.permute(0, 3, 1, 2)  # [B, 2, H, W]


class LabColorizationDataset(Dataset):
    def __init__(self, dataset_path, transform_gray=None, transform_color=None):
        """
        dataset_path debe tener dos directorios:
          - "imgs_gray": Imagenes en blanco y negro (input del modelo)
          - "imgs_color": Imagenes a color (output esperado)
        """
        self.dataset_path = dataset_path
        self.gray_dir = os.path.join(dataset_path, "imgs_gray")
        self.color_dir = os.path.join(dataset_path, "imgs_color")
        self.image_names = os.listdir(self.gray_dir)
        self.transform_gray = transform_gray
        self.transform_color = transform_color

    def __len__(self):
        return len(self.image_names)

    
    def __getitem__(self, idx):
        # Cargamos la imagen RGB y la blanco y negro
        gray_path = os.path.join(self.gray_dir, self.image_names[idx])
        color_path = os.path.join(self.color_dir, self.image_names[idx])

        color_img = Image.open(color_path).convert("RGB")
        # gray_img = Image.open(gray_path).convert("RGB")  # Grayscale

        # if self.transform_gray:
        #     gray_img = self.transform_gray(gray_img)
        if self.transform_color:
            color_img = self.transform_color(color_img)

        img_rgb = np.array(color_img).astype(np.float32)
        img_rgb = np.transpose(img_rgb, (1, 2, 0))

        # img_gray = np.array(gray_img).astype(np.float32)
        # img_gray = np.transpose(img_gray, (1, 2, 0))
        # Convert RGB -> Lab
        # skimage.color.rgb2lab expects float in [0,1]?
        img_lab = color.rgb2lab(img_rgb)
        # img_gray_lab = color.rgb2lab(img_gray)

        # Separamos la luminosidad L de (a, b)
        # La forma de LAB es (H, W, 3)
        L = img_lab[:, :, 0:1]  # (H, W, 1)
        ab = img_lab[:, :, 1:3] # (H, W, 2)

        L = L / 100.0 # Escalo L a [0,1]
        ab = (ab + 128) / 255.0  # Escalo ab a [0,1]
        # Convertimos a torch tensors -> (C, H, W)
        L_t = torch.from_numpy(L.transpose((2, 0, 1)))    # (1, H, W)
        ab_t = torch.from_numpy(ab.transpose((2, 0, 1)))  # (2, H, W)
        rgb_t = torch.from_numpy(img_rgb.transpose((2,0,1))) # (3, H, W)
        # print(f'L_t : {L_t.min()} {L_t.max()}, ab_t : {ab_t.min()} {ab_t.max()}, rgt_t : {rgb_t.min()} {rgb_t.max()}')

        Z = soft_encode(ab_t, ab_centroids, show_heatmap=False)  # Realizamos el soft-encoding
        print(f"Z shape: {Z.shape}")

        # Visualizar la predicción (soft decode)
        ab_pred = soft_decode(Z, ab_centroids, show_heatmap=False)  # Decodificamos la distribución

        
        ab_pred_img = ab_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

        plt.imshow(ab_pred_img)  # Visualiza la predicción final en el espacio ab
        plt.title("Predicción de colores ab después de soft_decode")
        plt.colorbar()
        plt.show()

        # L_t [0,1] - ab_t [0,1] - rgb_t [0,1]
        return L_t, ab_t, rgb_t


