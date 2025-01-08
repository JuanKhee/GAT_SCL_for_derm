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
from utils.graph_utils import batch_graphs
from tqdm import tqdm


class SkinDiseaseClassifier():
    def __init__(
            self,
            cnn_model,
            gat_model,
            epochs=10,
            batch_size=32,
            learning_rate=0.0001,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.Adam,
            output_dir='model_result'
    ):

        self.cnn_model = cnn_model
        self.gat_model = gat_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")

        self.cnn_model.to(self.device)
        self.gat_model.to(self.device)

        self.criterion = criterion
        self.optimizer = optimizer(
            list(self.cnn_model.parameters()) + list(self.gat_model.parameters()),
            lr=learning_rate
        )

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # else:
        #     raise Exception("Output Directory Exists, please ensure")

    def create_dataloader(
            self,
            train_root_path,
            test_root_path,
            seed=0,
            train_transform=None,
            test_transform=None,
            collate_fn=None
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
            shuffle=True,
            collate_fn=collate_fn
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

    def train_model(self):
        loss_progress = {}
        acc_progress = {}
        print(f'Total number of batches: {len(self.train_loader)}')
        cur_loss = None
        # Iterate x epochs over the train data
        self.cnn_model.train()
        self.gat_model.train()
        for epoch in range(self.epochs):
            print('current epoch: ', epoch)
            epoch_loss = []
            epoch_acc = []
            epoch_tracker = tqdm(self.train_loader)
            for batch in epoch_tracker:
                epoch_tracker.set_description(
                    f"loss: {np.average(epoch_loss) if len(epoch_loss) > 0 else None}; acc: {np.average(epoch_acc) if len(epoch_acc) > 0 else None}"
                )

                inputs, labels = batch
                cnn_inputs = torch.tensor([inp[0].numpy() for inp in inputs])
                # cnn_inputs = torch.tensor([batch[0][0][0].numpy()])
                gat_inputs = [inp[1] for inp in inputs]
                # gat_inputs = batch[0][0][1]
                gat_batch = (gat_inputs, labels)

                h, adj, src, tgt, Msrc, Mtgt, Mgraph, gat_labels = batch_graphs(gat_batch)
                h, adj, src, tgt, Msrc, Mtgt, Mgraph = map(
                    torch.from_numpy,
                    (h, adj, src, tgt, Msrc, Mtgt, Mgraph)
                )
                h = h.to(self.device)
                adj = adj.to(self.device)
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                Msrc = Msrc.to(self.device)
                Mtgt = Mtgt.to(self.device)
                Mgraph = Mgraph.to(self.device)
                gat_labels = gat_labels.to(self.device)

                cnn_inputs = cnn_inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                cnn_outputs = self.cnn_model(cnn_inputs)
                cnn_outputs = torch.nn.Sigmoid()(cnn_outputs)
                gat_outputs = self.gat_model(h, adj, src, tgt, Msrc, Mtgt, Mgraph)
                gat_outputs = torch.nn.Sigmoid()(gat_outputs)

                outputs = cnn_outputs * gat_outputs
                outputs = torch.nn.Softmax(dim=1)(outputs)

                # Labels are automatically one-hot-encoded
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                acc = np.average(outputs.max(1).indices.detach().cpu().numpy() == labels.detach().cpu().numpy())
                epoch_loss.append(loss.item())
                epoch_acc.append(acc)

            loss_progress[epoch] = np.average(epoch_loss)
            acc_progress[epoch] = np.average(epoch_acc)
            print(f'epoch loss: {np.average(epoch_loss)}')
            print(f'epoch acc : {np.average(epoch_acc)}')

            torch.save(self.cnn_model.state_dict(), os.path.join(self.output_dir, f'cnn_model_epoch{epoch}.pkl'))
            torch.save(self.gat_model.state_dict(), os.path.join(self.output_dir, f'gat_model_epoch{epoch}.pkl'))
            if cur_loss is None:
                print(f'current epoch has smallest loss value: epoch {epoch}')
                print('replacing best model file')
                torch.save(self.cnn_model.state_dict(), os.path.join(self.output_dir, f'cnn_best_model.pkl'))
                torch.save(self.gat_model.state_dict(), os.path.join(self.output_dir, f'gat_best_model.pkl'))
            else:
                if np.average(epoch_loss) < cur_loss:
                    print(f'current epoch has smallest loss value: epoch {epoch}')
                    print('replacing best model file')
                    torch.save(self.cnn_model.state_dict(), os.path.join(self.output_dir, f'cnn_best_model.pkl'))
                    torch.save(self.gat_model.state_dict(), os.path.join(self.output_dir, f'gat_best_model.pkl'))


        # torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model.pkl'))
        training_loss = pd.DataFrame(
            {'epoch': loss_progress.keys(), 'loss':loss_progress.values(), 'acc': acc_progress.values()}
        )
        training_loss.to_csv(os.path.join(self.output_dir, 'training_loss.csv'))

    def load_model(self):
        assert self.cnn_model is not None
        assert self.gat_model is not None
        self.cnn_model.load_state_dict(torch.load(os.path.join(self.output_dir, 'cnn_best_model.pkl')))
        self.gat_model.load_state_dict(torch.load(os.path.join(self.output_dir, 'gat_best_model.pkl')))

    def evaluate_model(self):
        assert self.cnn_model is not None
        assert self.gat_model is not None
        print(f'Test size: {len(self.test_dataset)}')
        all_labels = np.array([])
        all_outputs = np.array([])

        self.cnn_model.eval()
        self.gat_model.eval()
        for i, batch in enumerate(self.test_loader, 0):
            inputs, labels = batch
            cnn_inputs = torch.tensor([inp[0].numpy() for inp in inputs])
            # cnn_inputs = torch.tensor([batch[0][0][0].numpy()])
            gat_inputs = [inp[1] for inp in inputs]
            # gat_inputs = batch[0][0][1]
            gat_batch = (gat_inputs, labels)

            h, adj, src, tgt, Msrc, Mtgt, Mgraph, gat_labels = batch_graphs(gat_batch)
            h, adj, src, tgt, Msrc, Mtgt, Mgraph = map(
                torch.from_numpy,
                (h, adj, src, tgt, Msrc, Mtgt, Mgraph)
            )
            h = h.to(self.device)
            adj = adj.to(self.device)
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            Msrc = Msrc.to(self.device)
            Mtgt = Mtgt.to(self.device)
            Mgraph = Mgraph.to(self.device)
            gat_labels = gat_labels.to(self.device)

            cnn_inputs = cnn_inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            cnn_outputs = self.cnn_model(cnn_inputs)
            cnn_outputs = torch.nn.Sigmoid()(cnn_outputs)
            gat_outputs = self.gat_model(h, adj, src, tgt, Msrc, Mtgt, Mgraph)
            gat_outputs = torch.nn.Sigmoid()(gat_outputs)

            outputs = cnn_outputs * gat_outputs
            outputs = torch.nn.Softmax(dim=1)(outputs)

            print(outputs)
            print(outputs.size())
            outputs = outputs.max(1).indices.detach().cpu().numpy()
            print(outputs)
            print(labels)
            print(f"Batch {i} accuracy: ", (labels.detach().cpu().numpy() == outputs).sum() / len(labels))
            all_labels = np.concatenate((all_labels, labels), axis=None)
            all_outputs = np.concatenate((all_outputs, outputs), axis=None)
            print(f"Cumulative accuracy after batch: ", (all_labels == all_outputs).sum() / len(all_labels))
        print(classification_report(all_labels, all_outputs))


if __name__ == "__main__":
    from models.GAT_superpixel import GAT_image
    from utils.graph_utils import ImgToGraphTransform, graph_collate, ImageGraphDualTransform
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

    CNN_model = CNNModel()
    GAT_model = GAT_image(5,8)

    dev_classifier = SkinDiseaseClassifier(
        cnn_model=CNN_model,
        gat_model=GAT_model,
        epochs=2,
        batch_size=16,
        output_dir='dev_model_result_gatcnn'
    )
    dev_classifier.create_dataloader(
        train_root_path='dev_images/train',
        test_root_path='dev_images/test',
        train_transform=ImageGraphDualTransform(
            img_transform=transforms.Compose([
                transforms.RandomResizedCrop(size=(255, 255), scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.6678, 0.5298, 0.5245], std=[0.2232, 0.2030, 0.2146])
            ]),
            graph_transform=transforms.Compose([
                transforms.ToTensor(),
                ImgToGraphTransform(75)
            ])
        ),
        test_transform=ImageGraphDualTransform(
            img_transform=transforms.Compose([
                transforms.RandomResizedCrop(size=(255, 255), scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.6678, 0.5298, 0.5245], std=[0.2232, 0.2030, 0.2146])
            ]),
            graph_transform=transforms.Compose([
                transforms.ToTensor(),
                ImgToGraphTransform(75)
            ])
        ),
        collate_fn=graph_collate,
        seed=57
    )
    dev_classifier.train_model()
    dev_classifier.load_model()
    dev_classifier.evaluate_model()