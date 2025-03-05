import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from sklearn.model_selection import train_test_split

from nn.unet import UNet
from dataset.dataset_class import RecolorizationDataset

import time
import os

#############################################################
# Manipulación del dataset y preparación para entrenamiento #
#############################################################

# Transform para el dataset en blanco y negro
gray_transforms = transforms.Compose([
    transforms.Resize((254, 254), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    # If your grayscale images are in range [0,255], you might normalize to [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # adjust as needed
])

# Transform para las imagenes a color
color_transforms = transforms.Compose([
    transforms.Resize((254, 254), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # adjust as needed
])

dataset_path = "path/to/dataset"
gray_dir = os.path.join(dataset_path, "imgs_gray")

dataset_image_names = os.listdir(gray_dir)

# Dividir en train (70%) y temp (30%)
train_names, temp_names = train_test_split(dataset_image_names, test_size=0.3, random_state=42)
val_names, test_names = train_test_split(temp_names, test_size=0.5, random_state=42)

# Crear datasets (son todos iguales ya que despues los filtramos)
train_dataset = RecolorizationDataset("path/to/dataset", transform_gray=gray_transforms, transform_color=color_transforms)
val_dataset = RecolorizationDataset("path/to/dataset", transform_gray=gray_transforms, transform_color=color_transforms)
test_dataset = RecolorizationDataset("path/to/dataset", transform_gray=gray_transforms, transform_color=color_transforms)

# Filtrar imágenes para cada split
train_dataset.image_names = train_names
val_dataset.image_names = val_names
test_dataset.image_names = test_names

# Hiperparámetros
num_epochs = 10
batch_size = 4
learning_rate = 0.001
context_dim = 512 # dimension del vector de yolo, debería setearlo automaticamente, tambien podría llegar a ser un embedding de texto

# Crear datasets y dataloaders para entrenamiento y validación
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



######################################
# Definicion de la UNET y parámetros #
######################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=1, n_classes=3, context_dim=context_dim, bilinear=True)
model.to(device)

criterion = nn.L1Loss()  # Probar diferentes loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Creo que el Adam va a ir bien

# Directorio para guardar checkpoints del modelo
os.makedirs('checkpoints', exist_ok=True)
best_val_loss = float('inf')

#################
# Entrenamiento #
#################

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    start_time = time.time()
    
    for batch_idx, (img_gray, condition, img_color) in enumerate(train_loader):
        img_gray = img_gray.to(device)
        condition = condition.to(device)
        img_color = img_color.to(device)
        
        optimizer.zero_grad()
        outputs = model(img_gray, condition)
        loss = criterion(outputs, img_color)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * img_gray.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Validación
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for img_gray, condition, img_color in val_loader:
            img_gray = img_gray.to(device)
            condition = condition.to(device)
            img_color = img_color.to(device)
            
            outputs = model(img_gray, condition)
            loss = criterion(outputs, img_color)
            val_loss += loss.item() * img_gray.size(0)
    val_loss /= len(val_loader.dataset)
    
    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Tiempo: {elapsed:.2f}s")
    
    # Guardar el modelo si la pérdida de validación mejora
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = os.path.join("checkpoints", "best_model.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print("Modelo guardado en:", checkpoint_path)

