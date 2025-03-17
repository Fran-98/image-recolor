from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from network.misc import lab_to_rgb_batch
from network.imgs_manipulation import soft_decode
from torchviz import make_dot
from network.context import get_yolo_context


def quant_infer_and_display_lab(model, image_path, gamut_ab, input_size=(256, 256)):
    """
    Carga una imagen en escala de grises, la procesa, realiza inferencia en LAB usando el modelo entrenado
    y muestra la imagen colorizada.

    Parámetros:
      model: El modelo U-Net entrenado.
      image_path: Ruta a la imagen en escala de grises.
      gamut_ab: Mapeo de la paleta de 162 colores en el espacio ab, forma (162, 2).
      input_size: Tamaño de entrada deseado (ancho, alto).
    
    Retorna:
      output_pil: La imagen colorizada como PIL Image.
    """

    preprocess = transforms.Compose([
        # transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),  # Convierte a [0,1]
    ])

    # Cargar la imagen en escala de grises
    img_gray = Image.open(image_path).convert("L")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Procesar para obtener el canal L (forma: (1, 1, H, W))
    L_tensor = preprocess(img_gray).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():

        # El modelo ahora devuelve logits para 162 clases: (1, 162, H, W)
        pred_Z = model(L_tensor)

        pred_Z = torch.softmax(pred_Z, dim=1)

        # Permutar para que tenga forma [B, H, W, Q]
        pred_Z = pred_Z.permute(0, 2, 3, 1).contiguous()

        pred_ab = soft_decode(pred_Z, gamut_ab)
        
        print("Predicted indices stats: min =", pred_ab.min().item(),
              "max =", pred_ab.max().item(),
              "mean =", pred_ab.float().mean().item())
    
    # Convertir la imagen LAB (L y ab) a RGB
    colorized_rgb = lab_to_rgb_batch(L_tensor, pred_ab)  # Se espera (1, 3, H, W)
    
    # Convertir el tensor a imagen PIL para mostrarla
    output_pil = transforms.ToPILImage()(colorized_rgb.squeeze(0).cpu())

    # Cargar la imagen a color original (asumiendo estructura similar a entrenamiento)
    original_color_path = image_path.replace('imgs_gray', 'imgs_color')
    img_original = Image.open(original_color_path).convert("RGB")

    # Mostrar las imágenes: gris, colorizada y original
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(img_gray, cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("Original Grayscale")
    
    axs[1].imshow(output_pil)
    axs[1].axis("off")
    axs[1].set_title("Colorized Image")
    
    axs[2].imshow(img_original)
    axs[2].axis("off")
    axs[2].set_title("Original Color Image")
    
    plt.show()

    return output_pil
    
