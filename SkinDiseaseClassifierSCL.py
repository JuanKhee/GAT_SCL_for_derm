# %% Imports
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import copy
import pandas as pd 
import sys
from sklearn.metrics import classification_report
import cv2


class SkinDiseaseClassifier():
    def __init__(
            self,
            model,
            epochs=10,
            batch_size=32,
            learning_rate=0.0001,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.Adam,
            output_dir='model_result'
    ):

        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")

        self.model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer(model.parameters(), lr=learning_rate)
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            raise Exception("Output Directory Exists, please ensure")

    def create_dataloader(
            self,
            train_root_path,
            test_root_path,
            seed=0,
            train_transform=None,
            test_transform=None
    ):
        self.train_root_path = train_root_path
        self.test_root_path = test_root_path
        torch.manual_seed(seed)
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.Resize((255, 255)),
                transforms.ToTensor()
            ])
        if test_transform is None:
            test_transform = transforms.Compose([
                transforms.Resize((255, 255)),
                transforms.ToTensor()
            ])
        self.train_dataset = torchvision.datasets.ImageFolder(
            root=train_root_path,
            transform=train_transform
        )
        self.test_dataset = torchvision.datasets.ImageFolder(
            root=test_root_path,
            transform=test_transform
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def train_model(self):
        loss_progress = {}
        acc_progress = {}
        print(f'Total number of batches: {len(self.train_loader)}')
        # Iterate x epochs over the train data
        for epoch in range(self.epochs):
            epoch_loss = []
            epoch_acc = []
            for i, batch in enumerate(self.train_loader, 0):
                print(f'epoch {epoch}, batch {i}')
                inputs, labels = batch
                inputs = torch.cat([inputs[0], inputs[1]], dim=0)
                # print(inputs.shape)
                # cv2.imshow('0', inputs[0].permute(1,2,0).numpy())
                # cv2.waitKey(0)
                # cv2.imshow('1', inputs[1].permute(1,2,0).numpy())
                # cv2.waitKey(1)
                # cv2.imshow('2', inputs[2].permute(1,2,0).numpy())
                # cv2.waitKey(1)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                bsz = labels.shape[0]
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                f1, f2 = torch.split(outputs, [bsz, bsz], dim=0)
                outputs = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                # Labels are automatically one-hot-encoded
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                print(outputs.max(2).indices.detach().cpu().numpy()[:,0])
                print(labels)
                acc = np.average(outputs.max(2).indices.detach().cpu().numpy()[:,0] == labels.detach().cpu().numpy())
                print(f'  loss: {loss.item()}')
                print(f'  acc : {acc}')
                epoch_loss.append(loss.item())
                epoch_acc.append(acc)

            loss_progress[epoch] = np.average(epoch_loss)
            acc_progress[epoch] = np.average(epoch_acc)

        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model.pkl'))
        training_loss = pd.DataFrame(
            {'epoch': loss_progress.keys(), 'loss':loss_progress.values(),'acc': acc_progress.values()}
        )
        training_loss.to_csv(os.path.join(self.output_dir, 'training_loss.csv'))

    def load_model(self):
        assert self.model is not None
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, 'model.pkl')))

    def evaluate_model(self):
        assert self.model is not None
        print(f'Test size: {len(self.test_dataset)}')
        all_labels = np.array([])
        all_outputs = np.array([])

        for i, batch in enumerate(self.test_loader, 0):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.numpy()
            outputs = self.model(inputs)
            print(outputs)
            outputs = outputs.max(1).indices.detach().cpu().numpy()
            print(outputs)
            print(labels)
            print(f"Batch {i} accuracy: ", (labels == outputs).sum() / len(labels))
            all_labels = np.concatenate((all_labels, labels), axis=None)
            all_outputs = np.concatenate((all_outputs, outputs), axis=None)
            print(f"Cumulative accuracy after batch: ", (all_labels == all_outputs).sum() / len(all_labels))
        print(classification_report(all_labels, all_outputs))


if __name__ == "__main__":
    from losses.SupConLoss import SupConLoss
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


    vgg16_model = CNNModel()
    dev_classifier = SkinDiseaseClassifier(
        vgg16_model,
        epochs=2,
        batch_size=2,
        learning_rate=0.0001,
        output_dir='dev_model_result_vgg16_SCL2',
        criterion=SupConLoss()
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