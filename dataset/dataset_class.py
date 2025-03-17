from PIL import Image
from torch.utils.data import Dataset
import os
import torch
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
from network.colorfulness import get_gamut_colors_custom

gamut_colors = get_gamut_colors_custom(grid_size=10, L_value=50, tolerance=0.655)
ab_centroids = np.array([[c[3], c[4]] for c in gamut_colors])
ab_centroids = torch.tensor(ab_centroids, dtype=torch.float32).to('cuda')

def soft_encode_test(img_ab, ab_centroids, sigma=5, show_heatmap=False):
    img_ab = img_ab.permute(1, 2, 0).unsqueeze(0).unsqueeze(3).to('cuda')  # [1, H, W, 1, 2]

    img_ab = (img_ab * 254) - 127  # Escalar correctamente

    B, H, W, _, _ = img_ab.shape  # Corregir error en dimensiones
    Q = ab_centroids.shape[0]

    ab_centroids = ab_centroids.view(1, 1, 1, Q, 2)  # [1, 1, 1, Q, 2]

    # Calcular distancias cuadradas
    distances_sq = torch.sum((img_ab - ab_centroids) ** 2, dim=-1)  # [B, H, W, Q]

    # Encontrar los 5 vecinos más cercanos
    _, top5_indices = torch.topk(distances_sq, k=5, dim=3, largest=False)

    # Crear tensor Z de ceros
    Z = torch.zeros((B, H, W, Q), device=img_ab.device)

    # Obtener distancias y calcular pesos
    distances = torch.sqrt(torch.gather(distances_sq, dim=3, index=top5_indices))  
    weights = torch.exp(-distances**2 / (2 * sigma**2))
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8)

    # Aplicar a Z sin index_add_
    for b in range(B):
        for h in range(H):
            for w in range(W):
                Z[b, h, w, top5_indices[b, h, w]] = weights[b, h, w]

    # Normalizar para que sea distribución de probabilidad
    Z = Z / (Z.sum(dim=-1, keepdim=True) + 1e-8)

    return Z

def soft_decode_test(Z, ab_centroids, T=0.38, show_heatmap=False):
    Z_T = torch.exp(torch.log(Z + 1e-8) / T)  # Evitar log(0) con 1e-8
    Z_T = Z_T / Z_T.sum(dim=-1, keepdim=True)  # Renormalizar    

    # Obtener el valor esperado (annealed mean)
    ab_pred = torch.sum(Z_T.unsqueeze(-1) * ab_centroids.view(1, 1, 1, -1, 2), dim=3)
    if show_heatmap:
        heatmap = Z_T[0].sum(dim=2).detach().cpu().numpy()  # Sumar sobre la dimensión Q para visualizar
        plt.imshow(heatmap)
        plt.title("Soft Decode Heatmap")
        plt.colorbar()
        plt.show()

    return ab_pred.permute(0, 3, 1, 2) # [B, 2, H, W] 

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
    # img_ab viene normalizado de 0 a 1 y como B 2 H W,
    # ab_centroids viene de -127 a 123 [313, 2] Q 2
    # print(f' Encoder img_ab shape: {img_ab.shape} | ab_centroids shape: {ab_centroids.shape} | ab_centroids min: {ab_centroids.min()} | ab_centroids max: {ab_centroids.max()}')
    # img_ab = img_ab.permute(1, 2, 0).unsqueeze(0).unsqueeze(3).to('cuda')  # [1, H, W, 1, 2]

    img_ab = (img_ab * 254) - 127  # Escalar correctamente

    # B, H, W, _, _ = img_ab.shape  # Corregir error en dimensiones
    
    B, _, H, W = img_ab.shape
    Q = ab_centroids.shape[0]

    # Expandir dimensiones para broadcasting
    img_ab = img_ab.permute(0, 2, 3, 1).unsqueeze(3)  # [B, H, W, 1, 2]
    ab_centroids = ab_centroids.view(1, 1, 1, Q, 2)  # [1, 1, 1, Q, 2]
    # print(f'Encoder despues de permutar img_ab shape: {img_ab.shape} | ab_centroids shape: {ab_centroids.shape}')

    # Calcular distancias cuadradas
    distances_sq = torch.sum((img_ab - ab_centroids) ** 2, dim=4)  # [B, H, W, Q]

    # Encontrar los 5 vecinos más cercanos
    _, top5_indices = torch.topk(distances_sq, k=5, dim=3, largest=False)  # [B, H, W, 5]

    # Crear tensor Z de ceros
    Z = torch.zeros((B, H, W, Q), device=img_ab.device)

    # Obtener distancias y calcular pesos
    distances = torch.sqrt(torch.gather(distances_sq, dim=3, index=top5_indices))  
    weights = torch.exp(-distances**2 / (2 * sigma**2))
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8)

    # Aplicar a Z sin index_add_
    for b in range(B):
        for h in range(H):
            for w in range(W):
                Z[b, h, w, top5_indices[b, h, w]] = weights[b, h, w]

    # Normalizar para que sea distribución de probabilidad
    Z = Z / (Z.sum(dim=-1, keepdim=True) + 1e-8)

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

    # print(f"Valores de Z antes de log: {Z.min()} | {Z.max()} / Z shape: {Z.shape}")
    Z_T = torch.exp(torch.log(Z + 1e-8) / T)  # Evitar log(0) con 1e-8

    # Verificar la suma de Z_T antes de la normalización
    # print(f"Suma de Z_T antes de normalización: {Z_T.sum(dim=-1).min()} | {Z_T.sum(dim=-1).max()} / Z_T shape: {Z_T.shape}")

    # Renormalizar
    Z_T = Z_T / Z_T.sum(dim=-1, keepdim=True)


    # print(f"Suma de Z_T después de normalización: {Z_T.sum(dim=-1).min()} | {Z_T.sum(dim=-1).max()} / Z_T shape: {Z_T.shape}")
    # print(f'Z_T {Z_T.min()} | {Z_T.max()} / {Z_T.shape}')
    # Obtener el valor esperado (annealed mean)
    ab_pred = torch.sum(Z_T.unsqueeze(-1) * ab_centroids.view(1, 1, 1, -1, 2), dim=3) # B C H W
    # print(f'ab_pred shape: {ab_pred.shape}, ab_pred min: {ab_pred.min()}, ab_pred max: {ab_pred.max()}')
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

        if self.transform_color:
            color_img = self.transform_color(color_img)

        img_rgb = np.array(color_img).astype(np.float32)
        img_rgb = np.transpose(img_rgb, (1, 2, 0))

        img_lab = color.rgb2lab(img_rgb)
        L = img_lab[:, :, 0:1]  # (H, W, 1)
        ab = img_lab[:, :, 1:3]  # (H, W, 2)

        L = L / 100.0  # Escalo L a [0,1]
        ab = (ab + 128) / 255.0  # Escalo ab a [0,1]

        L_t = torch.from_numpy(L.transpose((2, 0, 1)))    # (1, H, W)
        ab_t = torch.from_numpy(ab.transpose((2, 0, 1)))  # (2, H, W)
        rgb_t = torch.from_numpy(img_rgb.transpose((2, 0, 1)))  # (3, H, W)

        # Z = soft_encode(ab_t.unsqueeze(0).to('cuda'), ab_centroids, show_heatmap=False)
        # # print(f'Salida de modelo {Z.min()} | {Z.max()} / {Z.shape}')

        # # Z = torch.softmax(Z, dim=1, out=Z)

        # # Visualizar la predicción (soft decode)
        # ab_pred = soft_decode(Z, ab_centroids, show_heatmap=False)

        # ab_pred_img = ab_pred.squeeze(0)  # Remove the batch dimension (only if batch size is 1)

        # # If the batch size is greater than 1, select the first image in the batch
        # if ab_pred_img.dim() == 4:
        #     ab_pred_img = ab_pred_img[0]  # Select the first image in the batch

        # # Now permute [2, 128, 128] -> [128, 128, 2] (for color channels a and b)
        # ab_pred_img = ab_pred_img.permute(1, 2, 0).cpu().numpy()


        # # Get the original L channel (already normalized to [0, 1])
        # L_channel = L_t.squeeze(0).cpu().numpy() * 100

        # # Convert back to Lab by combining the predicted ab channels with the original L channel
        # ab_pred_rgb = np.stack([L_channel, ab_pred_img[..., 0], ab_pred_img[..., 1]], axis=-1)  # [H, W, 3]

        # # Convert Lab back to RGB
        # ab_pred_rgb = color.lab2rgb(ab_pred_rgb)
    

        # Visualize the result
        # plt.imshow(ab_pred_rgb)
        # plt.title("Predicción de colores RGB después de soft_decode")
        # plt.colorbar()
        # plt.show()

        return L_t, ab_t, rgb_t
