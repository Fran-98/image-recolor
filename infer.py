from network.inference import infer_and_display_lab
from network.unet import UNet

import torch
   

checkpoint_path = r'checkpoints_3_30min_test\best_model.pth'
checkpoint_path = r'checkpoints\best_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(n_channels=1, n_classes=2, bilinear=True)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)


image_path = r"dataset\test_dataset\imgs_gray\00000069_(5).jpg"
# image_path = r"dataset\dataset_images\imgs_gray\00000001_(7).jpg"

infer_and_display_lab(model, image_path, input_size=(512,512))