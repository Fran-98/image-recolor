import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from sklearn.model_selection import train_test_split

from network.unet import UNet
from dataset.dataset_class import RecolorizationDataset

import matplotlib.pyplot as plt
import time
import os
import csv

from tqdm import tqdm
import shutil

def train_model(
        dataset_path: str,
        num_epochs: int,
        batch_size: int = 8,
        learning_rate: float = 0.0005,
        weight_decay: float = 0.00005
                ):
    
    ######################################
    # Definicion de la UNET y parámetros #
    ######################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=3, bilinear=True)
    model.to(device)

    criterion = nn.L1Loss()  # Probar diferentes loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Creo que el Adam va a ir bien
    #############################################################
    # Manipulación del dataset y preparación para entrenamiento #
    #############################################################

    # Transform para el dataset en blanco y negro
    gray_transforms = transforms.Compose([
        transforms.Resize((254, 254), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Transform para las imagenes a color
    color_transforms = transforms.Compose([
        transforms.Resize((254, 254), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    gray_dir = os.path.join(dataset_path, "imgs_gray")
    color_dir = os.path.join(dataset_path, "imgs_color")

    dataset_image_names = os.listdir(gray_dir)[:30]

    # Dividir en train (70%) y temp (30%)
    train_names, temp_names = train_test_split(dataset_image_names, test_size=0.10, random_state=42)
    val_names, test_names = train_test_split(temp_names, test_size=0.30, random_state=42)

    # Crear datasets (son todos iguales ya que despues los filtramos)
    train_dataset = RecolorizationDataset(dataset_path, transform_gray=gray_transforms, transform_color=color_transforms)
    val_dataset = RecolorizationDataset(dataset_path, transform_gray=gray_transforms, transform_color=color_transforms)
    test_dataset = RecolorizationDataset(dataset_path, transform_gray=gray_transforms, transform_color=color_transforms)

    # Filtrar imágenes para cada split
    train_dataset.image_names = train_names
    val_dataset.image_names = val_names
    test_dataset.image_names = test_names

    # Crear datasets y dataloaders para entrenamiento y validación
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    separar_test = False
    if separar_test:
        test_img_gray_dir = os.path.join(dataset_path, 'test_dataset', 'imgs_gray')
        test_img_color_dir = os.path.join(dataset_path, 'test_dataset', 'imgs_color')

        # Verificar si las carpetas existen, si no, crearlas
        os.makedirs(test_img_gray_dir, exist_ok=True)
        os.makedirs(test_img_color_dir, exist_ok=True)

        # Copiar las imágenes del test set en las nuevas carpetas
        for img_name in test_names:
            gray_img_path = os.path.join(gray_dir, img_name)
            color_img_path = os.path.join(color_dir, img_name)  # Assuming color images have the same name
            
            # Copiar las imágenes de la versión en blanco y negro y a color
            if os.path.exists(gray_img_path) and os.path.exists(color_img_path):
                shutil.copy(gray_img_path, os.path.join(test_img_gray_dir, img_name))
                shutil.copy(color_img_path, os.path.join(test_img_color_dir, img_name))
            else:
                print(f"Missing image pair for {img_name}, skipping...")

        print(f"Test dataset saved in {test_img_gray_dir} and {test_img_color_dir}")
        return
    # Directorio para guardar checkpoints del modelo
    os.makedirs('checkpoints', exist_ok=True)
    best_val_loss = float('inf')

    #################
    # Entrenamiento #
    #################
    train_losses = []
    val_losses = []
    stats_file = os.path.join('checkpoints', "training_stats.csv")

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch_idx, (img_gray, img_color) in tqdm(enumerate(train_loader)):
            img_gray = img_gray.to(device)
            img_color = img_color.to(device)

            optimizer.zero_grad()
            outputs = model(img_gray)
            loss = criterion(outputs, img_color)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * img_gray.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)  # Save the training loss

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img_gray, img_color in val_loader:
                img_gray = img_gray.to(device)
                img_color = img_color.to(device)

                outputs = model(img_gray)
                loss = criterion(outputs, img_color)
                val_loss += loss.item() * img_gray.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)  # Save the validation loss

        elapsed = time.time() - start_time

        # Guardar en el CSV
        with open(stats_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, val_loss, elapsed])

        checkpoint_path = os.path.join("checkpoints", f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Epoch {epoch+1}/{num_epochs}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Tiempo: {elapsed:.2f}s")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join("checkpoints", "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print("Modelo guardado en:", checkpoint_path)

    # After training, plot the losses
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    loss_plot_path = os.path.join('checkpoints', "training_loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.show()
