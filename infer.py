from network.inference import quant_infer_and_display_lab
from network.unet import UNetClasif

import numpy as np
import torch
from network.colorfulness import get_gamut_colors_custom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gamut_colors = get_gamut_colors_custom(grid_size=10, L_value=50, tolerance=0.655) # 313 valores dentro del gamut
gamut_ab = np.array([[c[3], c[4]] for c in gamut_colors])
gamut_ab = torch.tensor(gamut_ab, dtype=torch.float32).to(device)

checkpoint_path = r'checkpoints_imgnet_5_20k\best_model.pth'

model = UNetClasif(n_channels=1, n_classes=len(gamut_colors), bilinear=True)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)


image_path = r"dataset\dataset_256\imgs_gray\ILSVRC2012_val_00008305.JPEG" # Buenisima con diferentes objetos para testear
# image_path = r"dataset\dataset_256\imgs_gray\2017-02-16 08_21_05.jpg"
image_path = r"dataset\dataset_256\imgs_gray\2017-02-20 18_46_26.jpg"
# image_path = r"C:\Users\Fran\Downloads\descarga.jpg"

quant_infer_and_display_lab(model, image_path, gamut_ab, input_size=(256,256))