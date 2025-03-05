from PIL import Image
from torch.utils.data import Dataset
import os

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