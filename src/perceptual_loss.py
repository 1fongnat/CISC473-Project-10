import torch
import torch.nn as nn
from torchvision import models

class PerceptualLoss(nn.Module):
    """
    Calculates the LPIPS (Learned Perceptual Image Patch Similarity) metric.
    This uses a pre-trained network (like AlexNet) to extract features from
    two images and computes the distance between those features.
    """
    def __init__(self, net='alex', use_gpu=True):
        super(PerceptualLoss, self).__init__()
        # Check if GPU is available and user wants to use it
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        if net == 'alex':
            self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).features
        elif net == 'vgg':
            self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        else:
            raise ValueError(f"Network '{net}' not recognized for LPIPS.")

        self.model.eval()
        self.model.to(self.device) # Move model to the correct device

        for param in self.model.parameters():
            param.requires_grad = False
            
        # Normalization values for ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x, y):
        # Move input tensors to the correct device
        x_on_device = x.to(self.device)
        y_on_device = y.to(self.device)

        x_norm = self.normalize(x_on_device)
        y_norm = self.normalize(y_on_device)
        
        x_feats = self.model(x_norm)
        y_feats = self.model(y_norm)
        
        return nn.functional.mse_loss(x_feats, y_feats)