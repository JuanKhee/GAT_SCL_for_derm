import numpy as np
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


def gen_A_correlation(num_classes, t1, t2, adj_file):  ### checked
    import pickle
    _adj = pickle.load(open(adj_file, 'rb'))

    import numpy
    _adj = np.array(_adj)
    numpy.set_printoptions(threshold=np.inf)

    _adj[_adj >= t1] = 0  # t1 = 0.4
    _adj[_adj < t2] = 0  # t2 = 0.2
    _adj[(_adj >= t2) & (_adj < t1)] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    return _adj