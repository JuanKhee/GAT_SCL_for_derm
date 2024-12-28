import numpy as np
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from SkinDiseaseClassifierMTSCL import SkinDiseaseClassifier
from losses.loss_functions import SupConCELoss
from utils.supcon_utils import TwoCropTransform

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

print(torch.randint(5, (3,), dtype=torch.int64))
vgg16_model = CNNModel()
dev_classifier = SkinDiseaseClassifier(
    vgg16_model,
    epochs=2,
    batch_size=16,
    learning_rate=0.0001,
    output_dir='dev_model_result_vgg16_MTSCL3',
    criterion=SupConCELoss()
)

train_transform = TwoCropTransform(transforms.Compose([
    transforms.Resize(size=(255, 255)),
    transforms.RandomResizedCrop(size=(255, 255), scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()
]))

test_transform = transforms.Compose([
    transforms.Resize(size=(255, 255)),
    transforms.ToTensor()
])

dev_classifier.create_dataloader(
    train_root_path='dev_images/train',
    test_root_path='dev_images/test',
    seed=57,
    train_transform=train_transform,
    test_transform=test_transform
)
# print(dev_classifier.train_dataset[0][0][0].shape)
# print(dev_classifier.train_dataset[0][0][0].permute(1, 2, 0).shape)
# print(dev_classifier.train_dataset[0][0][0].permute(1, 2, 0).numpy().shape)
# cv2.imshow('img', dev_classifier.train_dataset[0][0][0].permute(1, 2, 0).numpy())
# cv2.waitKey(0)
# cv2.imshow('img', dev_classifier.train_dataset[0][0][1].permute(1, 2, 0).numpy())
# cv2.waitKey(0)
dev_classifier.train_model()
dev_classifier.load_model()
dev_classifier.evaluate_model()