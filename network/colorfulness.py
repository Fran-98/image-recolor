import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import KDTree
from PIL import Image

def lab_to_rgb_unclipped_scalar(L, a, b):
    """
    Convierte LAB (L en [0,100], a y b en valores típicos de [-128, 127]) a sRGB sin clipear.
    Devuelve un array [R, G, B] que puede tener valores fuera de [0,1] si el color está fuera del gamut.
    """
    # Conversión LAB -> XYZ (fórmula estándar)
    fy = (L + 16) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    epsilon = 216 / 24389.0  # ~0.008856
    kappa = 24389 / 27.0     # ~903.3

    fx3 = fx**3
    fy3 = fy**3
    fz3 = fz**3

    xr = fx3 if fx3 > epsilon else (116 * fx - 16) / kappa
    yr = fy3 if fy3 > epsilon else (116 * fy - 16) / kappa
    zr = fz3 if fz3 > epsilon else (116 * fz - 16) / kappa

    # Referencia D65
    X = xr * 0.95047
    Y = yr * 1.00000
    Z = zr * 1.08883

    # Conversión de XYZ a RGB lineal
    R_lin =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    G_lin = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    B_lin =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

    def gamma_correct(c):
        # Si c es negativo, se mantiene negativo (fuera del gamut)
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return (1 + 0.055) * (c ** (1/2.4)) - 0.055

    R = gamma_correct(R_lin)
    G = gamma_correct(G_lin)
    B = gamma_correct(B_lin)

    return np.array([R, G, B])

def get_gamut_colors_custom(grid_size=10, L_value=50, tolerance=0.01):
    """
    Genera los colores del gamut en el espacio LAB (float) usando la conversión sin clip.
    Se muestrean los valores de a y b en el rango [-110, 110] y se conserva
    el punto solo si, al convertir a RGB, todos sus canales están en [0,1] (con tolerancia).
    """
    a_vals = np.arange(-127, 127, grid_size)
    b_vals = np.arange(-127, 127, grid_size)
    gamut_colors = []
    for a in a_vals:
        for b in b_vals:
            rgb = lab_to_rgb_unclipped_scalar(L_value, a, b)
            if np.all(rgb >= 0 - tolerance) and np.all(rgb <= 1 + tolerance):
                gamut_colors.append((rgb[0], rgb[1], rgb[2], a, b))
    return gamut_colors

def plot_gamut_custom(gamut_colors, grid_size=10):
    """
    Visualiza la distribución de la paleta (gamut) en el espacio ab.
    El eje horizontal representa b* y el vertical, a*.
    """
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_aspect('equal')
    ax.set_xlabel('b*')
    ax.set_ylabel('a*')
    ax.set_title('Gamut en espacio ab (L fijo)')
    ax.set_xlim(-110, 110)
    ax.set_ylim(110, -110)
    ax.grid(True, linestyle='--', alpha=0.5)
    for r, g, b, a, b_val in gamut_colors:
        # Clipa los valores RGB para la visualización, pero internamente seguimos trabajando con los originales
        facecolor = tuple(np.clip(np.array([r, g, b]), 0, 1))
        rect = patches.Rectangle((b_val - grid_size/2, a - grid_size/2),
                                 grid_size, grid_size,
                                 facecolor=facecolor, edgecolor='none')
        ax.add_patch(rect)
    plt.show()

def quantize_image_custom(image_path, gamut_colors):
    """
    Cuantiza la imagen usando KDTree en el espacio ab (según nuestro gamut).
    Se preserva el canal L original de la imagen (obtenido con cv2) y se
    cuantizan los canales a y b usando la paleta generada.
    """
    # Cargar imagen y convertir a RGB en rango [0,1]
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img_rgb / 255.0

    # Preparar la paleta: extraemos los valores (a, b) de cada color del gamut
    gamut_ab = np.array([[c[3], c[4]] for c in gamut_colors])
    kdtree = KDTree(gamut_ab)

    # Convertir la imagen a LAB usando OpenCV (LAB en uint8: L [0,255], a y b [0,255] con centro 128)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    # Extraer y escalar a y b: de [0,255] a [-110,110]
    img_ab = img_lab[:, :, 1:].reshape(-1, 2) / 255.0 * 220 - 110

    # Buscar en el KDTree el vecino más cercano para cada píxel
    _, indices = kdtree.query(img_ab)
    quantized_colors_rgb = np.array([gamut_colors[i][:3] for i in indices])
    quantized_rgb = quantized_colors_rgb.reshape(img_array.shape)

    # Convertir los colores cuantizados (en RGB float) a LAB usando OpenCV
    # Se escala a [0,255] para la conversión
    quantized_rgb_uint8 = np.uint8(np.clip(quantized_rgb * 255, 0, 255))
    img_lab_reconstructed = cv2.cvtColor(quantized_rgb_uint8, cv2.COLOR_RGB2LAB)
    
    # Reemplazar el canal L original de la imagen
    original_L = img_lab[:, :, 0]
    img_lab_reconstructed[:, :, 0] = original_L

    # Reconvertir a RGB
    quantized_rgb_with_L = cv2.cvtColor(img_lab_reconstructed, cv2.COLOR_LAB2RGB)
    return Image.fromarray(quantized_rgb_with_L)

# --- Ejecución principal ---
if __name__ == '__main__':
    image_path = r"dataset\todas\00000001_(2).jpg"  # Actualiza con la ruta de tu imagen
    grid_size = 10
    tolerance = 0.655
    L_value = 50

    # Genera y muestra el gamut
    gamut_colors = get_gamut_colors_custom(grid_size=grid_size, L_value=L_value, tolerance=tolerance)
    print(f'Gamut range {np.min(gamut_colors)} | {np.max(gamut_colors)}')
    print("Número de colores en el gamut:", len(gamut_colors))
    plot_gamut_custom(gamut_colors, grid_size=grid_size)

    # # Cuantiza la imagen y la guarda/visualiza
    # quantized_image = quantize_image_custom(image_path, gamut_colors)
    # quantized_image.save("quantized_image_custom.png")
    # quantized_image.show()
