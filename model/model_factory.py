import torch 
from model.CNN.resnet import resnet50, resnet101, resnet152
from model.Transformers.VIT.vit import ViT
from model.Transformers.swin_transformers.models.swin_transformer import SwinTransformer
from model.Transformers.CMT.cmt import ConvolutionMeetVisionTransformers

class Regisiter(object):
    def __init__(self, name) -> None:
        super().__init__()
        self._name = name 
        self.obj = { }

    def setmodel(self, model_name, model):
        self.obj[model_name] = model 

    def getmodel(self, model_name):
        return self.obj[model_name]
        

ModelFactory = Regisiter("model")

ModelFactory.setmodel("R50", resnet50)
ModelFactory.setmodel("R101", resnet101)
ModelFactory.setmodel("R152", resnet152)
ModelFactory.setmodel("vit", ViT)
ModelFactory.setmodel("swin", SwinTransformer)
ModelFactory.setmodel("cmtti", ConvolutionMeetVisionTransformers)

