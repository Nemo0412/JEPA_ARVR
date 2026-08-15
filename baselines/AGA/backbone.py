import torch
from torch import nn
import torch.nn.functional as F
import timm

class SwinBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=False)
        self.model.avgpool = torch.nn.Identity()
    
    def forward(self, x):
        features = []
        for i in range(x.size(1)):
            xi = x[:, i]
            tmp = self.model.forward_features(xi).mean(1)
            features.append(tmp)
        return torch.stack(features, dim=1)