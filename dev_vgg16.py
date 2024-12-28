import numpy as np
import sys
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from SkinDiseaseClassifier import SkinDiseaseClassifier

np.set_printoptions(threshold=sys.maxsize)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)

        # Replace output layer according to our problem
        in_feats = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(in_feats, 8)

    def forward(self, x):
        x = self.vgg16(x)
        return x

vgg16_model = CNNModel()
dev_classifier = SkinDiseaseClassifier(
    vgg16_model,
    epochs=2,
    batch_size=32,
    learning_rate=0.0001,
    output_dir='dev_model_result_vgg16'
)

train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(255, 255), scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])

test_transform = transforms.Compose([
                transforms.Resize((255, 255)),
                transforms.ToTensor()
            ])

dev_classifier.create_dataloader(
    train_root_path='dev_images/train',
    test_root_path='dev_images/test',
    seed=57,
    train_transform=train_transform,
    test_transform=test_transform
)
dev_classifier.train_model()
dev_classifier.load_model()
dev_classifier.evaluate_model()