import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import backbones.davit as davit
import backbones.convnext_v2.convnext_v2 as convnext_v2
import backbones.mobilevit_v1 as mobilevit_v1
import backbones.mobilevit_v3 as mobilevit_v3
import backbones.mobilenet as mobilenet
from backbones.swin import SwinTransformer
from backbones.swinv2 import SwinTransformerV2





def get_backbone(backbone_name):
    if 'swin' in backbone_name:
        return SwinTransformerBackbone(name = backbone_name)
    elif 'davit' in backbone_name:
        return DaViT(name = backbone_name)
    elif 'convnext_v2' in backbone_name:
        return ConvNeXtV2(name = backbone_name)
    elif 'mobilevit_v1' in backbone_name:
        return MobileViT_V1(name = backbone_name)
    elif 'mobilevit_v3' in backbone_name:
        return MobileViT_V3(name = backbone_name)
    elif 'mobilenet' in backbone_name:
        return MobileNetV3(name = backbone_name)
    else:
        raise ValueError(f"Unsupported backbone name: {backbone_name}")
    

class MobileNetV3(nn.Module):
    def __init__(self, name = 'mobilevit_t'):
        super().__init__()
        assert name in ['mobilenet_v3_large', 'mobilenet_v3_small', 'mobilenet_v3_extra_large'], f"Unsupported MobileNetV3 model name: {name}"
        if name == 'mobilenet_v3_extra_large':
            self.model = mobilenet.mobilenetv3_extra_large()
        elif name == 'mobilenet_v3_large':
            self.model = mobilenet.mobilenetv3_large()
        elif name == 'mobilenet_v3_small':
            self.model = mobilenet.mobilenetv3_small()

    def forward(self, x):
        multiscale_features = self.model(x) 
        return multiscale_features
    
    def get_multiscale_feature_dim(self):
        return [f.shape for f in self.forward(torch.randn(1, 3, 112, 112))]
    

class MobileViT_V3(nn.Module):
    def __init__(self, name = 'mobilevit_v3_t'):
        super().__init__()
        assert name in ['mobilevit_v3_b', 'mobilevit_v3_s', 'mobilevit_v3_xs', 'mobilevit_v3_xxs'], f"Unsupported MobileViT model name: {name}"
        if name == 'mobilevit_v3_s':
            self.model = mobilevit_v3.mobilevit_s()
        elif name == 'mobilevit_v3_xs':
            self.model = mobilevit_v3.mobilevit_xs()
        elif name == 'mobilevit_v3_xxs':
            self.model = mobilevit_v3.mobilevit_xxs()
        elif name == 'mobilevit_v3_b':
            self.model = mobilevit_v3.mobilevit_b()


    def forward(self, x):
        multiscale_features = self.model(x)
        return multiscale_features


    def get_multiscale_feature_dim(self):
        return [f.shape for f in self.forward(torch.randn(1, 3, 112, 112))]


class MobileViT_V1(nn.Module):
    def __init__(self, name = 'mobilevit_t'):
        super().__init__()
        assert name in ['mobilevit_v1_b', 'mobilevit_v1_s', 'mobilevit_v1_xs', 'mobilevit_v1_xxs'], f"Unsupported MobileViT model name: {name}"
        if name == 'mobilevit_v1_s':
            self.model = mobilevit_v1.mobilevit_s()
        elif name == 'mobilevit_v1_xs':
            self.model = mobilevit_v1.mobilevit_xs()
        elif name == 'mobilevit_v1_xxs':
            self.model = mobilevit_v1.mobilevit_xxs()
        elif name == 'mobilevit_v1_b':
            self.model = mobilevit_v1.mobilevit_b()


    def forward(self, x):
        multiscale_features = self.model(x)
        return multiscale_features


    def get_multiscale_feature_dim(self):
        return [f.shape for f in self.forward(torch.randn(1, 3, 112, 112))]

    

class ConvNeXtV2(nn.Module):
    def __init__(self, name = 'convnext_v2_t'):
        super().__init__()

        assert name in ['convnext_v2_t', 'convnext_v2_s', 'convnext_v2_b'], f"Unsupported ConvNeXtV2 model name: {name}"
        if name == 'convnext_v2_t':
            self.model = convnext_v2.convnextv2_tiny()
        elif name == 'convnext_v2_s':
            self.model = convnext_v2.convnextv2_small()
        elif name == 'convnext_v2_b':
            self.model = convnext_v2.convnextv2_base()

    def forward(self, x):
        multiscale_features = self.model(x) 
        return multiscale_features

    def get_multiscale_feature_dim(self):
        return [f.shape for f in self.forward(torch.randn(1, 3, 112, 112))]

        
        
class DaViT(nn.Module):
    def __init__(self, name = 'davit_t'):
        super().__init__()
        assert name in ['davit_t', 'davit_t_224', 'davit_s', 'davit_b','davit_b_224'], f"Unsupported DaViT model name: {name}"
        if name == 'davit_t':
            self.model = davit.DaViT_tiny()
        elif name == 'davit_t_224':
            self.model = davit.DaViT_tiny_224()
        elif name == 'davit_s':
            self.model = davit.DaViT_small()
        elif name == 'davit_b':
            self.model = davit.DaViT_base()
        elif name == 'davit_b_224':
            self.model = davit.DaViT_base_224()
    
    def forward(self, x):
        multiscale_features = self.model(x) 
        return multiscale_features

    def get_multiscale_feature_dim(self):
        return [f.shape for f in self.forward(torch.randn(1, 3, 112, 112))]


class SwinTransformerBackbone(nn.Module):
    def __init__(self, name='swin_t'):
        super().__init__()

        assert name in ['swin_t', 'swin_s', 'swin_b', 'swin_v2_t', 'swin_v2_s', 'swin_v2_b'], f"Unsupported Swin Transformer model name: {name}"    
        
        if name == 'swin_t':
            swin_embedding_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
        elif name == 'swin_s':
            swin_embedding_dim = 96
            depths = [2, 2, 18, 2]
            num_heads = [3, 6, 12, 24]
        elif name == 'swin_b':
            swin_embedding_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
        elif name == 'swin_v2_t':
            swin_embedding_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
        elif name == 'swin_v2_s':
            swin_embedding_dim = 96
            depths = [2, 2, 18, 2]
            num_heads = [3, 6, 12, 24]
        elif name == 'swin_v2_b':
            swin_embedding_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]

        if 'v2' in name:
            self.model = SwinTransformerV2(
                depths=depths,
                embed_dim = swin_embedding_dim,
                num_heads=num_heads
            )
        else:
            self.model = SwinTransformer(
                depths=depths,
                embed_dim = swin_embedding_dim,
                num_heads=num_heads
            )
            

    def forward(self, x):
        multiscale_features  = self.model(x)
        return multiscale_features
    
    def get_multiscale_feature_dim(self):
        return [f.shape for f in self.forward(torch.randn(1, 3, 112, 112))]


