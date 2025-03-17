import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from sklearn.model_selection import train_test_split

from network.unet import UNetClasif
from dataset.dataset_class import LabColorizationDataset
from network.misc import lab_to_rgb_batch, ConditionalResize
from network.inference import quant_infer_and_display_lab

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import csv

from tqdm import tqdm
import shutil

from network.telegram_notif import send_image, send_text
from network.losses import total_variation_loss, VGGPerceptualLoss, gradient_loss

from network.colorfulness import get_gamut_colors_custom
from network.imgs_manipulation import get_class_weights, soft_encode, soft_decode

import random

SEED = 13

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm(2).item() ** 2
    return total_norm ** 0.5  # Raíz cuadrada para obtener norma L2

def train_model(
        dataset_path: str,
        output_path: str = 'checkpoints',
        n_samples: int = None,
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        lambda_perceptual: float = 0.1,
        lambda_tv: float = 0.05,
        lambda_grad: float = 10,
        pre_trained: str = None,
        imgs_size: int = 256,
                ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")
    ######################################
    # Definicion de la UNET y parámetros #
    ######################################

    # Verificar e iniciar directorios
    if pre_trained:
        root_folder = pre_trained.replace(pre_trained.split('/')[-1],'')
    else:
        os.makedirs(output_path, exist_ok=True)
        root_folder = output_path

    os.makedirs(os.path.join(root_folder, "debug"), exist_ok=True)

    gamut_colors = get_gamut_colors_custom(grid_size=10, L_value=50, tolerance=0.655) # 313 valores dentro del gamut
    gamut_ab = np.array([[c[3], c[4]] for c in gamut_colors])
    gamut_ab = torch.tensor(gamut_ab, dtype=torch.float32).to(device)
    
    class_weights = get_class_weights(dataset_path, gamut_ab, imgs_size=imgs_size)
    # class_weights = None
    print(f"Class Weights: min={class_weights.min()}, max={class_weights.max()}, mean={class_weights.mean()}")

    print(f'Gamut range {gamut_ab.min()} | {gamut_ab.max()}')
    print("Número de colores en el gamut:", len(gamut_colors))
    
    model = UNetClasif(n_channels=1, n_classes=len(gamut_colors), bilinear=True)
    model.initialize_weights() # Cuando lo uso nunca logran converger los colores

    model.to(device)

    # Loss que compara pixel a pixel

    #criterion_pixel = nn.L1Loss()
    # criterion_pixel = nn.MSELoss() # Obtuve mejores resultados con mse
    # criterion_pixel = nn.CrossEntropyLoss()
    criterion_pixel = nn.CrossEntropyLoss(weight=class_weights).to(device)

    criterion_perceptual = VGGPerceptualLoss(resize=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Creo que el Adam va a ir bien
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False,)

    if pre_trained:
        model.load_state_dict(torch.load(pre_trained, map_location=device))
        optimizer_checkpoint = torch.load(pre_trained.replace('model_epoch', 'optimizer'), map_location=device)
        optimizer.load_state_dict(optimizer_checkpoint['optimizer_state_dict'])
    #############################################################
    # Manipulación del dataset y preparación para entrenamiento #
    #############################################################

    # Transform para el dataset en blanco y negro
    val_transforms = transforms.Compose([
        ConditionalResize(imgs_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])


    train_transform = transforms.Compose([
    ConditionalResize(imgs_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
    ])

    gray_dir = os.path.join(dataset_path, "imgs_gray")
    color_dir = os.path.join(dataset_path, "imgs_color")

    dataset_image_names = sorted(os.listdir(gray_dir))

    if n_samples:
        dataset_image_names = random.sample(dataset_image_names, min(n_samples, len(dataset_image_names)))

    # Dividir en train (95%) y temp (5%)
    train_names, temp_names = train_test_split(dataset_image_names, test_size=0.05, random_state=SEED)
    val_names, test_names = train_test_split(temp_names, test_size=0.30, random_state=SEED)

    # Crear datasets (son todos iguales ya que despues los filtramos)

    train_dataset = LabColorizationDataset(dataset_path, transform_gray=train_transform, transform_color=train_transform)
    val_dataset = LabColorizationDataset(dataset_path, transform_gray=val_transforms, transform_color=val_transforms)
    test_dataset = LabColorizationDataset(dataset_path, transform_gray=val_transforms, transform_color=val_transforms)    


    # Filtrar imágenes para cada split
    train_dataset.image_names = train_names
    val_dataset.image_names = val_names
    test_dataset.image_names = test_names

    # print(train_names[:3])
    # Crear datasets y dataloaders para entrenamiento y validación
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Hay {train_dataset.__len__()} imagenes en el train')
    print(f'Hay {val_dataset.__len__()} imagenes en el val')

    if not os.path.isdir(os.path.join(dataset_path, 'test_dataset')):
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
                shutil.move(gray_img_path, os.path.join(test_img_gray_dir, img_name))
                shutil.move(color_img_path, os.path.join(test_img_color_dir, img_name))
            else:
                print(f"Par faltante para {img_name}")

        print(f"Test dataset guardado {test_img_gray_dir} y {test_img_color_dir}")
        # return

    

    #################
    # Entrenamiento #
    #################

    # Iniciamos variables que vamos a utilizar
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    restart_epoch = 0
    type_training = 'Nuevo Entrenamiento'
    stats_file = os.path.join(root_folder, "training_stats.csv")

    if pre_trained:
        type_training = 'Reanudando entrenamiento'
        with open(stats_file) as csvfile:
            reader = csv.reader(csvfile)
            
            for row in reader:
                train_losses.append(float(row[1]))
                val_losses.append(float(row[2]))
                last_row = row
            restart_epoch = int(last_row[0])
    else:
        if os.path.exists(stats_file):
            os.remove(stats_file)
            
    send_text(f"""{type_training}
              Epocas: {num_epochs}
              LR: {learning_rate}
              Weight decay: {weight_decay}
              Batch size: {batch_size}
              Lambda perceptual: {lambda_perceptual}
              Lambda TV: {lambda_tv}
              Lambda gradient: {lambda_grad}
              """)
    
    ###########################
    # Loop de entrenamiento!! #
    ###########################
    for epoch in tqdm(range(num_epochs), initial=restart_epoch, desc="Epoch", position=0):
        epoch += restart_epoch
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (img_L, img_ab, img_rgb) in enumerate(tqdm(train_loader, desc="Train Batch", position=1)):

            img_L = img_L.to(device)       # (B,1,H,W)
            img_ab = img_ab.to(device)     # (B,2,H,W)
            img_rgb = img_rgb.to(device)   # (B,3,H,W)

            optimizer.zero_grad()

            ab_denormalized = (img_ab * 254) - 127
            Z_gt = soft_encode(ab_denormalized, gamut_ab)  # [B, H, W, Q]
            
            Z_gt = Z_gt.permute(0, 3, 1, 2).contiguous()  # [B, Q, H, W]  <-- Importante para CrossEntropyLoss

            # Z_gt lo tengo comprobado que esta bien!
            #print(f'Z_gt {Z_gt.min()} | {Z_gt.max()} / {Z_gt.shape}')

            # print(f'Entrada de modelo {model_input.min()} | {model_input.max()}')
            pred_Z = model(img_L)  # (B,C,H,W)

            # Acá hago la perdida de entropia cruzada para luego aplicar los pesos
            loss_pixel = criterion_pixel(pred_Z, Z_gt) # Perdida pixel-wise en LAB (solo en ab)

            # print(f"Suma de pred_Z antes de softmax: {pred_Z.sum(dim=-1).min()} | {pred_Z.sum(dim=-1).max()}")
            # print(f"Antes de softmax: min={pred_Z.min()} max={pred_Z.max()}")
            # print(f"Pred_Z logits - Mean: {pred_Z.mean().item()}, Std: {pred_Z.std().item()}")
            # print(f"Min: {pred_Z.min().item()}, Max: {pred_Z.max().item()}")

            # pred_Z = (pred_Z - pred_Z.mean(dim=1, keepdim=True)) / (pred_Z.std(dim=1, keepdim=True) + 1e-6)
            pred_Z = torch.softmax(pred_Z, dim=1)
            # print(pred_Z)
            # print(f"Índice más probable después de softmax: {pred_Z.argmax(dim=1).unique()}")
            # sum_softmax = pred_Z.sum(dim=1)
            # print(f"Suma por pixel después de softmax - Min: {sum_softmax.min().item()}, Max: {sum_softmax.max().item()}")
            # avg_pred_Z = pred_Z.mean(dim=1)  # Promedio de clases en cada pixel
            # print(f"Promedio por pixel después de softmax - Min: {avg_pred_Z.min().item()}, Max: {avg_pred_Z.max().item()}")
            # print(f'Salida de modelo despues Softmax mi  {pred_Z.min()} | max {pred_Z.max()} / {pred_Z.shape}')
            # print(f"Suma de pred_Z: {pred_Z.sum(dim=-1).min()} | {pred_Z.sum(dim=-1).max()}")

            # print(f'Salida de modelo softmax {pred_Z.min()} | {pred_Z.max()} / {pred_Z.shape}')
            # print(f'Suma de pred_Z en bins: {pred_Z.sum(dim=1).mean()}')
            # Permutar para que tenga forma [B, H, W, Q]
            pred_Z = pred_Z.permute(0, 2, 3, 1).contiguous()

            pred_ab = soft_decode(pred_Z, gamut_ab) # De acá para abajo tambien funciona todo

            # print(f'AB pred {pred_ab.min()} | {pred_ab.max()} / {pred_ab.shape}')

            pred_rgb = lab_to_rgb_batch(img_L, pred_ab) # Creamos la imagen RGB
            
            loss_percep = criterion_perceptual(pred_rgb, img_rgb) # Perdida perceptual

            # loss_tv = total_variation_loss(pred_ab) # Perdida por variacion (nos permite suavizar la salida)

            # loss_grad = gradient_loss(pred_rgb, img_rgb)

            print("Pixel loss:", loss_pixel.item())
            print(f"Perceptual loss {loss_percep.item()} -> {loss_percep.item() * lambda_perceptual}", )
            # print(f"Loss TV {loss_tv.item()} -> {loss_tv.item() * lambda_tv}", )
            # print(f"Loss gradient {loss_grad.item()} -> {loss_grad.item() * lambda_grad}")

            # Mezclamos las perdidas
            loss = loss_pixel + lambda_perceptual * loss_percep
            loss.backward() # Backpropagation

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            train_loss += loss.item() * img_L.size(0)

            if batch_idx == 0:
                # Guardamos algunas predicciones del training para ver como avanza el modelo
                for i in range(min(pred_rgb.size(0), 2)):
                    debug_img = pred_rgb[i].detach().cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
                    debug_path = os.path.join(root_folder, "debug", f"train_epoch_{epoch+1}_{i}.png")
                    plt.imsave(debug_path, debug_img)
                    send_image(debug_path, f'Training epoch {epoch+1}') # Manda por telegram la imagen
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: {param.grad.abs().mean().item():.6f}")
                grad_norm = get_grad_norm(model)
                print(f"Epoch {epoch}: Grad Norm = {grad_norm:.6f}")
        train_loss /= len(train_loader)
        
        train_losses.append(train_loss)

        # Validación
        model.eval()
        val_loss = 0.0
        val_loss_pixel = 0.0
        val_loss_percep = 0.0
        val_loss_tv = 0.0
        val_loss_grad = 0.0
        first = True
        with torch.no_grad():
            for img_L, img_ab, img_rgb in val_loader:
                img_L = img_L.to(device)
                img_ab = img_ab.to(device)
                img_rgb = img_rgb.to(device)

                ab_denormalized = (img_ab * 254) - 127
                Z_gt = soft_encode(ab_denormalized, gamut_ab)  # [B, H, W, Q]
                Z_gt = Z_gt.permute(0, 3, 1, 2).contiguous()  # [B, Q, H, W]  <-- Importante para CrossEntropyLoss
                Z_gt = Z_gt.argmax(dim=1).long() # Ahora tiene forma [B, H, W]

                pred_Z = model(img_L)
                
                loss_pixel = criterion_pixel(pred_Z, Z_gt) # Perdida pixel-wise en LAB (solo en ab)

                pred_Z = (pred_Z - pred_Z.mean(dim=1, keepdim=True)) / (pred_Z.std(dim=1, keepdim=True) + 1e-6)
                pred_Z = torch.softmax(pred_Z, dim=1)

                # print(f'Salida de modelo Validation {pred_Z.min()} | {pred_Z.max()} / {pred_Z.shape}')
                # print(f"Suma de pred_Z: {pred_Z.sum(dim=-1).min()} | {pred_Z.sum(dim=-1).max()}")
                
                # Permutar para que tenga forma [B, H, W, Q]
                pred_Z = pred_Z.permute(0, 2, 3, 1).contiguous()

                pred_ab = soft_decode(pred_Z, gamut_ab)

                pred_rgb = lab_to_rgb_batch(img_L, pred_ab) # Creamos la imagen RGB
                
                loss_percep = criterion_perceptual(pred_rgb, img_rgb) # Perdida perceptual

                # loss_tv = total_variation_loss(pred_ab) # Perdida por variacion (nos permite suavizar la salida)

                # loss_grad = gradient_loss(pred_rgb, img_rgb)

                loss_total = loss_pixel + lambda_perceptual * loss_percep

                val_loss += loss_total.item() * img_L.size(0)
                val_loss_pixel += loss_pixel.item() * img_L.size(0)
                val_loss_percep += loss_percep.item() * img_L.size(0)
                # val_loss_tv += loss_tv.item() * img_L.size(0)
                # val_loss_grad += loss_grad.item() * img_L.size(0)

                if first:
                    first = False
                    # Guardamos algunas predicciones del training para ver como avanza el modelo
                    for i in range(min(pred_rgb.size(0), 2)):
                        debug_path = os.path.join(root_folder, "debug", f"validation_epoch{epoch+1}_{i}.png")
                        plt.imsave(debug_path, pred_rgb[i].detach().cpu().numpy().transpose(1, 2, 0))
                        send_image(debug_path, f'Validation epoch {epoch+1}')

        val_loss /= len(val_loader)
        val_loss_pixel /= len(val_loader)
        val_loss_percep /= len(val_loader)
        val_loss_tv /= len(val_loader)
        val_loss_grad /= len(val_loader)

        val_losses.append(val_loss)
        # print("Validation Loss: {:.4f}".format(val_loss))
        # print(" - Pixel Loss: {:.4f}".format(val_loss_pixel))
        # print(" - Perceptual Loss: {:.4f}".format(val_loss_percep))
        # print(" - TV Loss: {:.4f}".format(val_loss_tv))
        # print(" - Gradient Loss: {:.4f}".format(val_loss_grad))

        elapsed = time.time() - start_time

        scheduler.step(val_loss)

        # Guardar en el CSV
        with open(stats_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, val_loss, elapsed])

        checkpoint_path = os.path.join(root_folder, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        checkpoint_optimizer_path = os.path.join(root_folder, f"optimizer_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_losses,
            'val_loss_history': val_losses,
            }, checkpoint_optimizer_path)
        
        send_text(f"Epoch {epoch+1}/{num_epochs}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Tiempo: {elapsed:.2f}s")
        print(f"Epoch {epoch+1}/{num_epochs}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Tiempo: {elapsed:.2f}s")

        # Guardamos la mejor validacion como best_model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(root_folder, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            send_text(f"Mejor modelo guardado en epoca: {epoch+1}")
            print("Modelo guardado en:", checkpoint_path)

        # Despues de cada epoch guardamos el grafico de loss
        plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
        plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        loss_plot_path = os.path.join(root_folder, "training_loss_plot.png")
        plt.savefig(loss_plot_path)
        plt.close()
        send_image(loss_plot_path, f'Epoch {epoch+1}')

    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    loss_plot_path = os.path.join(root_folder, "training_loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.show()

    send_text('Entrenamiento finalizado')
    image_path = f"dataset\dataset_images\test_dataset\imgs_gray\{test_names[0]}"
    quant_infer_and_display_lab(model, image_path, gamut_ab, input_size=(imgs_size,imgs_size))