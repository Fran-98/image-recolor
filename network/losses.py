import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """"""
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features

        self.slice1 = vgg[:4] # relu1_2
        self.slice2 = vgg[4:9] # relu2_2
        self.slice3 = vgg[9:16] # relu3_3
        self.slice4 = vgg[16:23] # relu4_3
        for param in self.parameters():
            param.requires_grad = False
        self.resize = resize

        # Normalizacion
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1))

    def forward(self, x, y):
        """
        x, y: RGB [0,1] (B, 3, H, W)
        """
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        if self.resize:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = nn.functional.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        
        # recorremos la red
        x1 = self.slice1(x)
        y1 = self.slice1(y)

        x2 = self.slice2(x1)
        y2 = self.slice2(y1)

        x3 = self.slice3(x2)
        y3 = self.slice3(y2)

        x4 = self.slice4(x3)
        y4 = self.slice4(y3)

        # Computamos el loss
        loss = 0
        loss += nn.functional.l1_loss(x1, y1)
        loss += nn.functional.l1_loss(x2, y2)
        loss += nn.functional.l1_loss(x3, y3)
        loss += nn.functional.l1_loss(x4, y4)
        return loss
    
def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
    pred_dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]

    target_dx = target[:, :, :, :-1] - target[:, :, :, 1:]
    target_dy = target[:, :, :-1, :] - target[:, :, 1:, :]

    loss = torch.mean(torch.abs(pred_dx - target_dx)) + torch.mean(torch.abs(pred_dy - target_dy))
    return loss

def total_variation_loss(x):
    return torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + \
           torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))

