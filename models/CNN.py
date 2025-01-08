# %% Imports
import torch.nn as nn
import torchvision.models as models

class CNNModel(nn.Module):
    def __init__(self, output_dim, model_name='vgg16'):
        super(CNNModel, self).__init__()
        assert model_name in ('vgg16', 'resnet50', 'resnet152')

        if model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, output_dim)
            self.fc = nn.Identity()

        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
            self.fc = nn.Linear(num_features, output_dim)

        if model_name == 'resnet152':
            self.model = models.resnet152(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
            self.fc = nn.Linear(num_features, output_dim)

    def forward(self, x):
        x = self.model(x)
        out = self.fc(x)
        return out