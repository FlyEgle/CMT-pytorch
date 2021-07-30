import torch
import torch.nn as nn 
from model.CNN.resnet import resnet50


class R50Features(nn.Module):
    def __init__(self, imagenet_pretrain):
        super(R50Features, self).__init__()
        self.model = resnet50(pretrained=False, num_classes=1000)
        state_dict = torch.load(imagenet_pretrain, map_location="cpu")['state_dict']
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x 




        
