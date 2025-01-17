import torch
import torchvision.models as models

def load_model(model_name):
    if model_name == 'resnet18':
        return models.resnet18(pretrained=True)
    if model_name == 'resnet34':
        return models.resnet34(pretrained=True)
    if model_name == 'resnet50':
        return models.resnet50(pretrained=True)
    if model_name == 'resnet101':
        return models.resnet101(pretrained=True)
    if model_name == 'resnet152':
        return models.resnet152(pretrained=True)
    if model_name == 'resnext50_32x4d':
        return models.resnext50_32x4d(pretrained=True)
    if model_name == 'resnext101_32x8d':
        return models.resnext101_32x8d(pretrained=True)
    if model_name == 'wide_resnet50_2':
        return models.wide_resnet50_2(pretrained=True)
    if model_name == 'wide_resnet101_2':
        return models.wide_resnet101_2(pretrained=True)
    if model_name == 'resnet18_swsl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_swsl')
    if model_name == 'resnet50_swsl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
    if model_name == 'resnext50_32x4d_swsl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
    if model_name == 'resnext101_32x4d_swsl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_swsl')
    if model_name == 'resnext101_32x8d_swsl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_swsl')
    if model_name == 'resnext101_32x16d_swsl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl')
    if model_name == 'resnet18_ssl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_ssl')
    if model_name == 'resnet50_ssl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_ssl')
    if model_name == 'resnext50_32x4d_ssl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
    if model_name == 'resnext101_32x4d_ssl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_ssl')
    if model_name == 'resnext101_32x8d_ssl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_ssl')
    if model_name == 'resnext101_32x16d_ssl':
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_ssl')
    return None