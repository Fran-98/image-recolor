import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset.dataset_class import LabColorizationDataset
from network.misc import ConditionalResize
from tqdm import tqdm
import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian_weights(distances, sigma=5):
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    return weights / np.sum(weights)  # Normalizar para que sumen 1

def plot_ab_distribution(ab_centroids, distribution, title='Distribución en ab'):
    """
    Hace un scatter plot de la distribución de probabilidad en el espacio ab.

    Args:
        ab_centroids: Tensor shape (Q, 2) con los centroides (a, b).
        distribution: Tensor shape (Q,) con la probabilidad p(q).
        title: Título del gráfico.
    """
    # Convertir a CPU y numpy
    ab_centroids = ab_centroids.cpu().numpy()
    dist_np = distribution.cpu().numpy()

    # Separar coordenadas a y b
    a_vals = ab_centroids[:, 0]
    b_vals = ab_centroids[:, 1]

    # Usamos el log para realzar la densidad en colores saturados
    dist_log = np.log(dist_np + 1e-8)

    plt.figure(figsize=(6, 5))
    # Nota: si prefieres eje horizontal = b y vertical = a, invierte el orden
    sc = plt.scatter(b_vals, a_vals, c=dist_log, s=8, cmap='jet')
    plt.colorbar(sc, label='log(prob)')
    plt.xlabel('b')
    plt.ylabel('a')
    plt.title(title)
    plt.show()

def calculate_class_weights(dataset_path, 
                            ab_centroids, 
                            sigma=5, 
                            lambda_=0.5, 
                            batch_size=64, 
                            img_size=128, 
                            device='cuda',
                            plot_distributions=True):
    """
    Calcula los pesos para el re-balanceo de clases de forma vectorizada y
    opcionalmente grafica la distribución empírica y la suavizada.
    """

    num_bins = ab_centroids.shape[0]
    p = torch.zeros(num_bins, device=device)

    # Transformaciones
    gray_transforms = transforms.Compose([
        ConditionalResize(img_size),
        transforms.ToTensor(),
    ])
    color_transforms = transforms.Compose([
        ConditionalResize(img_size),
        transforms.ToTensor(),
    ])

    # Crear dataset y DataLoader
    temp_dataset = LabColorizationDataset(
        dataset_path,
        transform_gray=gray_transforms,
        transform_color=color_transforms,
    )
    # Asegurarnos de que lea todas las imágenes de "imgs_gray"
    gray_dir = os.path.join(dataset_path, "imgs_gray")
    temp_dataset.image_names = os.listdir(gray_dir)

    # Evitar problemas con DataLoader en Windows
    mp.set_start_method('spawn', force=True)
    temp_loader = DataLoader(temp_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=8, 
                             pin_memory=True)

    # Precalcula (ab_centroids) en GPU para distancias
    ab_centroids = ab_centroids.to(device)  # (Q,2)

    # Recorremos el dataset para contar frecuencias
    for _, img_ab, _ in tqdm(temp_loader, desc="Calculando histograma"):
        img_ab = img_ab.to(device)  # [B, 2, H, W]
        B, _, H, W = img_ab.shape

        # Reordenar a [B*H*W, 2]
        img_ab = img_ab.permute(0, 2, 3, 1).reshape(-1, 2)

        # Muestreo aleatorio: la mitad de los pixeles
        num_pixels = img_ab.shape[0]
        num_sample = int(num_pixels * 0.5)
        indices = torch.randint(0, num_pixels, (num_sample,), device=device)
        img_ab = img_ab[indices]

        # Desnormalizar: si tu img_ab venía en [0,1], pasa a [-128,128]
        img_ab = (img_ab * 256) - 128

        # Distancia al cuadrado a cada centro: [num_sample, Q]
        # Broadcasting => (N,1,2) - (1,Q,2) => sum(...) => (N,Q)
        distances_sq = torch.sum((img_ab.unsqueeze(1) - ab_centroids.unsqueeze(0)) ** 2, dim=2)

        # Bin más cercano
        closest_bins = torch.argmin(distances_sq, dim=1)

        # Actualizar histograma p
        p.index_add_(0, closest_bins, torch.ones_like(closest_bins, dtype=torch.float32, device=device))

    # Normalizar p para convertirlo en distribución
    p = p / p.sum()

    # --- Suavizado con gaussiana ---
    # Pasamos a CPU para usar gaussian_filter de scipy
    p_cpu = p.cpu().numpy()
    pe = gaussian_filter(p_cpu, sigma=sigma)

    # Mezclamos con distribución uniforme
    pe = (1 - lambda_) * pe + (lambda_ / num_bins)

    # Convertimos a tensor en GPU
    pe_t = torch.tensor(pe, device=device, dtype=torch.float32)

    # Calculamos w = 1 / pe
    w = 1.0 / (pe_t + 1e-8)

    # Normalizamos para que sum_q p(q)*w(q) = 1
    # (E[w] = 1, en la notación del paper)
    normalizer = torch.sum(p * w)
    w = w / (normalizer + 1e-8)

    # # Clampeamos, si quieres
    # w = torch.clamp(w, max=10.0)
    # # Ojo: si el clamp modifica mucho, a veces se renormaliza otra vez.

    # OPCIONAL: graficar la distribución empírica y/o suavizada
    if plot_distributions:
        from matplotlib import pyplot as plt
        # Plot de la distribución empírica p
        plot_ab_distribution(ab_centroids, p, "Distribución empírica p")

        # Plot de la distribución suavizada pe
        plot_ab_distribution(ab_centroids, pe_t, "Distribución suavizada p_e")

        # Plot de los pesos w (si te interesa verlos)
        plot_ab_distribution(ab_centroids, w, "Pesos w (rebalance)")

    return w

def get_class_weights(dataset_path, ab_centroids, sigma=5, lambda_=0.5, batch_size=64, imgs_size=128):
    if os.path.exists(os.path.join(dataset_path, 'class_weights.pt')):
        print(f"Cargando pesos desde {os.path.join(dataset_path, 'class_weights.pt')}")
        weights = torch.load(os.path.join(dataset_path, 'class_weights.pt'), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        print("Calculando pesos de clase...")
        weights = calculate_class_weights(dataset_path, ab_centroids, sigma=sigma, lambda_=lambda_, batch_size=batch_size, img_size=imgs_size)
        torch.save(weights, os.path.join(dataset_path, 'class_weights.pt'))
        print(f"Pesos guardados en {os.path.join(dataset_path, 'class_weights.pt')}")
    return weights

def soft_encode(img_ab, ab_centroids, sigma=5):
    """
    Realiza el soft-encoding de los valores ab a un tensor de probabilidades Z,
    pero usando operaciones vectorizadas (sin bucles triple anidado).
    
    Args:
        img_ab: Tensor de valores ab [B, 2, H, W] en el rango [-128, 128] aprox.
        ab_centroids: Tensor [Q, 2] con los centroides de cada bin.
        sigma: desviación estándar para el kernel Gaussiano.
        
    Returns:
        Tensor Z [B, H, W, Q], con la distribución de probabilidad soft-encoded.
    """
    B, _, H, W = img_ab.shape
    Q = ab_centroids.shape[0]

    # 1) Reordenar img_ab para que tenga forma [B, H, W, 2]
    img_ab = img_ab.permute(0, 2, 3, 1).contiguous()  # (B,H,W,2)
    
    # 2) Expandir centroids a [1,1,1,Q,2] para broadcasting
    ab_centroids = ab_centroids.view(1, 1, 1, Q, 2)  # (1,1,1,Q,2)
    
    # 3) Calcular distancias al cuadrado en el espacio ab
    #    diff tendrá forma (B,H,W,Q,2), luego sumamos en dim=4 => (B,H,W,Q)
    diff = img_ab.unsqueeze(3) - ab_centroids  # (B,H,W,Q,2)
    distances_sq = (diff ** 2).sum(dim=4)      # (B,H,W,Q)

    # 4) Hallar los 5 vecinos más cercanos (top-5 con menor distancia)
    #    top5_indices: (B,H,W,5), top5_distances: (B,H,W,5)
    top5_distances, top5_indices = torch.topk(distances_sq, k=5, dim=3, largest=False)

    # 5) Convertir distancias al kernel Gaussiano y normalizar entre esos 5
    top5_distances = top5_distances.sqrt()  # (B,H,W,5)
    weights = torch.exp(-top5_distances**2 / (2 * sigma**2))  # (B,H,W,5)
    weights = weights / (weights.sum(dim=3, keepdim=True) + 1e-8)  # normaliza en dim=3

    # 6) Construir Z con scatter_
    #    Z es un tensor (B,H,W,Q), y en la dimensión Q queremos "depositar"
    #    los pesos de cada píxel en sus 5 índices más cercanos.
    Z = torch.zeros((B, H, W, Q), device=img_ab.device)
    Z.scatter_(3, top5_indices, weights)  # en la dimensión 3 se ponen los 5 pesos

    # 7) (Opcional) Normalizar nuevamente si se desea que la suma en Q sea 1 exacta.
    #    En teoría, si sólo asignamos pesos a esos 5 bins, la suma ya debería ser 1.
    #    Pero puedes hacerlo por seguridad:
    Z = Z / (Z.sum(dim=3, keepdim=True) + 1e-8)

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
