from PIL import Image
from torch.utils.data import Dataset
import os

import torch
import numpy as np
from skimage import color

class RecolorizationDataset(Dataset):
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
        gray_path = os.path.join(self.gray_dir, self.image_names[idx])
        color_path = os.path.join(self.color_dir, self.image_names[idx])
        
        gray_img = Image.open(gray_path).convert("L")  # Grayscale
        color_img = Image.open(color_path).convert("RGB")
        
        if self.transform_gray:
            gray_img = self.transform_gray(gray_img)
        if self.transform_color:
            color_img = self.transform_color(color_img)
            
        return gray_img, color_img
    

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
        gray_img = Image.open(gray_path).convert("RGB")  # Grayscale

        if self.transform_gray:
            gray_img = self.transform_gray(gray_img)
        if self.transform_color:
            color_img = self.transform_color(color_img)

        # Convertimos a [0,1]
        img_rgb = np.array(color_img).astype(np.float32) #/ 255.0
        img_rgb = np.transpose(img_rgb, (1, 2, 0))

        img_gray = np.array(gray_img).astype(np.float32) #/ 255.0
        img_gray = np.transpose(img_gray, (1, 2, 0))
        # Convert RGB -> Lab
        # skimage.color.rgb2lab expects float in [0,1]
        img_lab = color.rgb2lab(img_rgb)
        img_gray_lab = color.rgb2lab(img_gray)

        # Separamos la luminosidad L de (a, b)
        # La forma de LAB es (H, W, 3)
        # L = img_gray_lab[:, :, 0:1]  # (H, W, 1)
        L = img_lab[:, :, 0:1]  # (H, W, 1)
        ab = img_lab[:, :, 1:3] # (H, W, 2)

        L = L / 100.0 # Escalo L a [0,1]
        ab = (ab + 128) / 255.0  # Escalo ab a [0,1]
        # Convertimos a torch tensors -> (C, H, W)
        L_t = torch.from_numpy(L.transpose((2, 0, 1)))    # (1, H, W)
        ab_t = torch.from_numpy(ab.transpose((2, 0, 1)))  # (2, H, W)
        rgb_t = torch.from_numpy(img_rgb.transpose((2,0,1))) # (3, H, W)
        return L_t, ab_t, rgb_t
    
