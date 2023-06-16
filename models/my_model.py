"""
Example model. 

Author: Jinhui Yi
Date: 2023.06.01
"""
import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.num_classes = cfg.num_classes
        # self.model_name = 'resnet50'
        self.model_name = 'swin_v2_s'
        assert self.model_name in models.list_models()
        
        print("Loading pretrained: ", self.model_name)
        self.model = getattr(models, self.model_name)(weights='DEFAULT')
        # self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes) # replace the final FC layer (ResNet50)
        self.model.head = nn.Linear(self.model.head.in_features, self.num_classes) # replace the final FC layer (swin_v2_s)

    def forward(self, x):
        return self.model(x)