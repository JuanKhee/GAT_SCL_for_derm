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
            transform=None
    ):
        self.train_root_path = train_root_path
        self.test_root_path = test_root_path
        torch.manual_seed(seed)
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((255, 255)),
                transforms.ToTensor()
            ])
        self.train_dataset = torchvision.datasets.ImageFolder(
            root=train_root_path,
            transform=transform
        )
        self.test_dataset = torchvision.datasets.ImageFolder(
            root=test_root_path,
            transform=transform
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
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # Labels are automatically one-hot-encoded
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                acc = np.average(outputs.max(1).indices.detach().cpu().numpy() == labels.detach().cpu().numpy())
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
            print(outputs.size())
            outputs = outputs.max(1).indices.detach().cpu().numpy()
            print(outputs)
            print(f"Batch {i} accuracy: ", (labels == outputs).sum() / len(labels))
            all_labels = np.concatenate((all_labels, labels), axis=None)
            all_outputs = np.concatenate((all_outputs, outputs), axis=None)
            print(f"Cumulative accuracy after batch: ", (all_labels == all_outputs).sum() / len(all_labels))
        print(classification_report(all_labels, all_outputs))


if __name__ == "__main__":
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
    dev_classifier = SkinDiseaseClassifier(vgg16_model, epochs=2, output_dir='dev_model_result')
    dev_classifier.create_dataloader(
        train_root_path='dev_images/train',
        test_root_path='dev_images/test',
        seed=57
    )
    dev_classifier.train_model()
    dev_classifier.load_model()
    dev_classifier.evaluate_model()