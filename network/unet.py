import torch
import torch.nn as nn

import torch.nn.functional as F

from network.context import get_yolo_context, num_classes

########################
# Bloques para la UNET #
########################

# En vez de definir uno por uno los elementos de la red, creamos las clases con los bloques necesarios para que luego sea
# mas sencilla la definicion

class DoubleConv(nn.Module):
    """Bloque de dos convoluciones seguidas de BatchNorm y ReLU."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class BottleNeck(nn.Module):
    """Bloque de dos convoluciones seguidas de BatchNorm y ReLU."""
    def __init__(self, in_channels, out_channels):
        super(BottleNeck, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            
        )
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Bloque de downsampling con max pooling seguido de DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class AttentionGate(nn.Module):
    """
    Puerta de atención que filtra la información de la skip connection.
    Recibe características del encoder (x) y la señal de gating (g) del decoder.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Up(nn.Module):
    """
    Bloque de upsampling que aplica una puerta de atención a la skip connection.
    Realiza upsampling, aplica atención, concatena y luego procesa con DoubleConv.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.attention = AttentionGate(F_g=in_channels//2, F_l=in_channels//2, F_int=in_channels//4)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x, skip_connection):
        x = self.up(x)
        # Ajustar tamaño
        if x.size() != skip_connection.size():
            diffY = skip_connection.size()[2] - x.size()[2]
            diffX = skip_connection.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        skip_connection = self.attention(skip_connection, x)
        x = torch.cat([skip_connection, x], dim=1)
        return self.conv(x)

class FiLM(nn.Module):
    """
    Módulo FiLM que modula las activaciones en función de una condición.
    Genera parámetros gamma y beta para escalar y desplazar las features.

    FiLM: Feature-wise Linear Modulation
    https://arxiv.org/pdf/1709.07871
    """
    def __init__(self, num_features, context_dim):
        super(FiLM, self).__init__()
        self.film_generator = nn.Linear(context_dim, num_features * 2)
        
    def forward(self, features, condition):
        gamma_beta = self.film_generator(condition)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)  # Forma: (batch, C, 1, 1)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return features * gamma + beta

################################
# Definicion de la UNET clasif #
################################

class UNetClasif(nn.Module):
    """
    U-Net modificada para recolorización con colores quantizados, que incorpora:
    - Atención en las skip connections.
    - Modulación condicional con FiLM en el bottleneck.

    Parámetros:
    - n_channels: número de canales de entrada (por ejemplo, 1 para imágenes en escala de grises).
    - n_classes: número de canales de salida (por ejemplo, 3 para imagen en color RGB).
    - context_dim: dimensión del vector de contexto salido de yolo.
    - bilinear: si se utiliza upsampling bilineal o convolución transpuesta.
    """
    def __init__(self, n_channels, n_classes, bilinear=True, dropout_rate=0.1):
        super(UNetClasif, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        base_channels = 128
        
        # Encoder
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # Bottleneck
        self.bottleneck = BottleNeck(base_channels * 16 // factor, base_channels * 16 // factor)
        # self.bottleneck = DoubleConv(base_channels * 16 // factor, base_channels * 16 // factor)
        # self.bottleneck = DoubleConv(base_channels * 32 // factor, base_channels * 32 // factor)

        self.dropout = nn.Dropout2d(dropout_rate)

        self.film = FiLM(num_features=base_channels * 16 // factor, context_dim=num_classes)
        # self.film = FiLM(num_features=base_channels * 32 // factor, context_dim=num_classes)
        
        # Decoder
        # self.up0 = Up(base_channels * 32, base_channels * 16 // factor, bilinear)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = nn.Sequential(
            nn.Conv2d(base_channels, n_classes, kernel_size=1),
            )
        
    def forward(self, x):
        """Recibe la imagen x"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x6 = self.down5(x5)

        # Bottleneck con modulación FiLM
        x_bottleneck = self.bottleneck(x5)
        x_bottleneck = self.dropout(x_bottleneck)
        x_bottleneck = self.film(x_bottleneck, get_yolo_context(x))
        #Conexión residual
        x_bottleneck = x_bottleneck + x5

        
        # Decoder con skip connections y attention gates
        x = self.up1(x_bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)


        
        logits = self.outc(x)
        return logits
    
    def initialize_weights(self):
        """
        Inicializa los pesos del modelo:
        - Conv2d y ConvTranspose2d: Kaiming Normal con mode='fan_in'.
        - Si la capa usa LeakyReLU, ajusta `nonlinearity='leaky_relu'`.
        - Si la capa es parte del bottleneck, usa inicialización ortogonal.
        - BatchNorm2d: Inicializa la escala en 1 y el sesgo en 0.
        - Linear: Xavier Normal.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Verifica si la activación es LeakyReLU
                nonlinearity = 'leaky_relu' if isinstance(getattr(m, 'activation', None), nn.LeakyReLU) else 'relu'
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=nonlinearity)

                # Si la capa tiene bias y no hay BatchNorm después, inicializa en 0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                # Inicialización de BatchNorm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                # Inicialización de capas lineales con Xavier Normal
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # Inicialización ortogonal en la capa bottleneck
            if isinstance(m, nn.Conv2d) and 'bottleneck' in m.__class__.__name__.lower():
                nn.init.orthogonal_(m.weight)