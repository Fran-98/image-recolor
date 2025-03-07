import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from sklearn.model_selection import train_test_split

from network.unet import UNet
from dataset.dataset_class import LabColorizationDataset
from network.misc import lab_to_rgb_batch
from network.inference import infer_and_display_lab

import matplotlib.pyplot as plt
import time
import os
import csv

from tqdm import tqdm
import shutil

from network.telegram_notif import send_image, send_text
from network.losses import total_variation_loss, VGGPerceptualLoss

def lr_lambda(current_step):
    warmup_steps = 10
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    else:
        return 1.0

def train_model(
        dataset_path: str,
        n_samples: int = None,
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-6,
        lambda_perceptual: float = 0.1,
        lambda_tv: float = 0.05,
        pre_trained: str = None,
                ):
    
    ######################################
    # Definicion de la UNET y parámetros #
    ######################################


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=2, bilinear=True)
    # model.initialize_weights() # Cuando lo uso nunca logran converger los colores
    if pre_trained:
        model.load_state_dict(torch.load(pre_trained, map_location=device))
    model.to(device)

    # Loss que compara pixel a pixel

    #criterion_pixel = nn.L1Loss()
    criterion_pixel = nn.MSELoss() # Obtuve mejores resultados con mse

    criterion_perceptual = VGGPerceptualLoss(resize=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Creo que el Adam va a ir bien
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True,)
    #############################################################
    # Manipulación del dataset y preparación para entrenamiento #
    #############################################################

    # Transform para el dataset en blanco y negro
    gray_transforms = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Transform para las imagenes a color
    color_transforms = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    gray_dir = os.path.join(dataset_path, "imgs_gray")
    color_dir = os.path.join(dataset_path, "imgs_color")

    if n_samples:
        dataset_image_names = os.listdir(gray_dir)[:n_samples]
    else:
        dataset_image_names = os.listdir(gray_dir)

    # Dividir en train (70%) y temp (30%)
    train_names, temp_names = train_test_split(dataset_image_names, test_size=0.10, random_state=42)
    val_names, test_names = train_test_split(temp_names, test_size=0.30, random_state=42)

    # Crear datasets (son todos iguales ya que despues los filtramos)

    train_dataset = LabColorizationDataset(dataset_path, transform_gray=gray_transforms, transform_color=color_transforms)
    val_dataset = LabColorizationDataset(dataset_path, transform_gray=gray_transforms, transform_color=color_transforms)
    test_dataset = LabColorizationDataset(dataset_path, transform_gray=gray_transforms, transform_color=color_transforms)    


    # Filtrar imágenes para cada split
    train_dataset.image_names = train_names#[:1]
    val_dataset.image_names = val_names
    test_dataset.image_names = test_names

    # print(train_names[:3])
    # Crear datasets y dataloaders para entrenamiento y validación
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    separar_test = True
    if separar_test:
        test_img_gray_dir = os.path.join(dataset_path, 'test_dataset', 'imgs_gray')
        test_img_color_dir = os.path.join(dataset_path, 'test_dataset', 'imgs_color')

        # Verificar si las carpetas existen, si no, crearlas
        os.makedirs(test_img_gray_dir, exist_ok=True)
        os.makedirs(test_img_color_dir, exist_ok=True)

        # Copiar las imágenes del test set en las nuevas carpetas
        for img_name in test_names:
            gray_img_path = os.path.join(gray_dir, img_name)
            color_img_path = os.path.join(color_dir, img_name)
            
            # Copiar las imágenes de la versión en blanco y negro y a color
            if os.path.exists(gray_img_path) and os.path.exists(color_img_path):
                shutil.copy(gray_img_path, os.path.join(test_img_gray_dir, img_name))
                shutil.copy(color_img_path, os.path.join(test_img_color_dir, img_name))
            else:
                print(f"Par faltante para {img_name}")

        print(f"Test dataset guarado {test_img_gray_dir} y {test_img_color_dir}")
        # return

    # Directorio para guardar checkpoints del modelo
    os.makedirs('checkpoints', exist_ok=True)
    best_val_loss = float('inf')

    #################
    # Entrenamiento #
    #################
    train_losses = []
    val_losses = []
    restart_epoch = 0
    stats_file = os.path.join('checkpoints', "training_stats.csv")
    type_training = 'Nuevo Entrenamiento'
    if pre_trained:
        stats_file = pre_trained.replace(pre_trained.split('/')[-1], "training_stats.csv")
        type_training = 'Reanudando entrenamiento'
        with open(stats_file) as csvfile:
            reader = csv.reader(csvfile)
            
            for row in reader:
                train_losses.append(float(row[1]))
                val_losses.append(float(row[2]))
                last_row = row
            restart_epoch = int(last_row[-1][0])

    
    send_text(f"""{type_training}
              Epocas: {num_epochs}
              LR: {learning_rate}
              Weight decay: {weight_decay}
              Batch size: {batch_size}
              Lambda perceptual: {lambda_perceptual}
              Lambda TV: {lambda_tv}
              """)
    
    ###########################
    # Loop de entrenamiento!! #
    ###########################
    for epoch in tqdm(range(num_epochs), initial=restart_epoch, desc="Epoch"):
        epoch += restart_epoch
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (img_L, img_ab, img_rgb) in enumerate(tqdm(train_loader, desc="Train Batch")):
            img_L = img_L.to(device)       # (B,1,H,W)
            img_ab = img_ab.to(device)     # (B,2,H,W)
            img_rgb = img_rgb.to(device)   # (B,3,H,W)

            optimizer.zero_grad()

            # Forward: la red predice ab (2 canales) dada L (1 canal)
            pred_ab = model(img_L)  # (B,2,H,W)
            # print(f"Output channel values:\nRed channel mean={pred_ab[:,0].mean()}, Green channel mean={pred_ab[:,1].mean()}")
            
            loss_pixel = criterion_pixel(pred_ab, img_ab) # Perdida pixel-wise en LAB (solo en ab)

            pred_rgb = lab_to_rgb_batch(img_L, pred_ab) # Creamos la imagen RGB
            
            loss_percep = criterion_perceptual(pred_rgb, img_rgb) # Perdida perceptual

            loss_tv = total_variation_loss(pred_ab) # Perdida por variacion (nos permite suavizar la salida)
        
            # print("Pixel loss:", loss_pixel.item())
            # print(f"Perceptual loss {loss_percep.item()} -> {loss_percep.item() * lambda_perceptual}", )
            # print(f"Loss TV {loss_tv.item()} -> {loss_tv.item() * lambda_tv}", )

            # Mezclamos las perdidas
            loss = loss_pixel + lambda_perceptual * loss_percep + loss_tv * lambda_tv
            loss.backward() # Backpropagation

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Clipping a los gradientes 
            optimizer.step()

            train_loss += loss.item() * img_L.size(0)

            if batch_idx == 0:
                # Guardamos algunas predicciones del training para ver como avanza el modelo
                for i in range(min(pred_rgb.size(0), 2)):
                    debug_img = pred_rgb[i].detach().cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
                    debug_path = "debug/debug_pred_rgb_epoch{}_sample{}.png".format(epoch+1, i)
                    plt.imsave(debug_path, debug_img)
                    send_image(debug_path, f'Training epoch {epoch}') # Manda por telegram la imagen

        train_loss /= len(train_loader.dataset)
        
        train_losses.append(train_loss)

        # Validación
        model.eval()
        val_loss = 0.0
        val_loss_pixel = 0.0
        val_loss_percep = 0.0
        val_loss_tv = 0.0
        first = True
        with torch.no_grad():
            for img_L, img_ab, img_rgb in val_loader:
                img_L = img_L.to(device)
                img_ab = img_ab.to(device)
                img_rgb = img_rgb.to(device)
                
                pred_ab = model(img_L)
                loss_pixel = criterion_pixel(pred_ab, img_ab)
                pred_rgb = lab_to_rgb_batch(img_L, pred_ab)
                loss_percep = criterion_perceptual(pred_rgb, img_rgb)
                loss_tv = total_variation_loss(pred_ab)
                if first:
                    debug_path = "debug/validation_epoch{}.png".format(epoch+1)
                    plt.imsave(debug_path, pred_rgb[0].detach().cpu().numpy().transpose(1, 2, 0))
                    # print("Saved debug image:", debug_path)
                    send_image(debug_path, f'Validation epoch {epoch+1}')
                loss_total = loss_pixel + lambda_perceptual * loss_percep + lambda_tv * loss_tv
                
                val_loss += loss_total.item() * img_L.size(0)
                val_loss_pixel += loss_pixel.item() * img_L.size(0)
                val_loss_percep += loss_percep.item() * img_L.size(0)
                val_loss_tv += loss_tv.item() * img_L.size(0)

        val_loss /= len(val_loader.dataset)
        val_loss_pixel /= len(val_loader.dataset)
        val_loss_percep /= len(val_loader.dataset)
        val_loss_tv /= len(val_loader.dataset)

        val_losses.append(val_loss)
        # print("Validation Loss: {:.4f}".format(val_loss))
        # print(" - Pixel Loss: {:.4f}".format(val_loss_pixel))
        # print(" - Perceptual Loss: {:.4f}".format(val_loss_percep))
        # print(" - TV Loss: {:.4f}".format(val_loss_tv))

        elapsed = time.time() - start_time

        scheduler.step(val_loss)

        # Guardar en el CSV
        with open(stats_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, val_loss, elapsed])

        checkpoint_path = os.path.join("checkpoints", f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        send_text(f"Epoch {epoch+1}/{num_epochs}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Tiempo: {elapsed:.2f}s")
        print(f"Epoch {epoch+1}/{num_epochs}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Tiempo: {elapsed:.2f}s")

        # Guardamos la mejor validacion como best_model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join("checkpoints", "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print("Modelo guardado en:", checkpoint_path)

        # Despues de cada epoch guardamos el grafico de loss
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        loss_plot_path = os.path.join('checkpoints', "training_loss_plot.png")
        plt.savefig(loss_plot_path)


    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    loss_plot_path = os.path.join('checkpoints', "training_loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.show()

    send_text('Entrenamiento finalizado')
    image_path = f"dataset\dataset_images\imgs_gray/{train_names[1]}"
    infer_and_display_lab(model, image_path, input_size=(512,512))