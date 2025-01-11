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
from tqdm import tqdm
from utils.data_utils import compute_mean_std

class SkinDiseaseClassifier():
    def __init__(
            self,
            model,
            epochs=10,
            batch_size=32,
            learning_rate=0.0001,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.Adam,
            output_dir='model_result',
            num_workers=0,
            pin_memory=False,
            train_mean=[0.6678, 0.5298, 0.5245],
            train_std=[0.2232, 0.2030, 0.2146]
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
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_mean = train_mean
        self.train_std = train_std

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
        self.train_transform = train_transform
        self.test_transform = test_transform
        torch.manual_seed(seed)
        if train_transform is None:
            self.train_transform = [
                transforms.Resize((255, 255)),
                transforms.ToTensor()
            ]
        if test_transform is None:
            self.test_transform = [
                transforms.Resize((255, 255)),
                transforms.ToTensor()
            ]
        self.train_dataset = torchvision.datasets.ImageFolder(
            root=train_root_path,
            transform=transforms.Compose(self.train_transform),
        )

        self.test_dataset = torchvision.datasets.ImageFolder(
            root=test_root_path,
            transform=transforms.Compose(self.test_transform)
        )
        self.train_dataset.transform = transforms.Compose(
            self.train_transform + [transforms.Normalize(mean=self.train_mean, std=self.train_mean)]
        )
        self.test_dataset.transform = transforms.Compose(
            self.test_transform + [transforms.Normalize(mean=self.train_mean, std=self.train_mean)]
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        self.val_dataset = None
        self.val_loader = None

    def cross_validate(self, k=2, seed=0, cv_suffix=''):

        assert 100 % k == 0
        torch.manual_seed(seed)
        k_datasets = torch.utils.data.random_split(self.train_dataset, [1/k]*k)
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'model_common_init_state.pkl'))
        for i, dataset in enumerate(k_datasets):
            print(f'fold {i}/{k}')
            self.model.load_state_dict(torch.load(os.path.join(self.output_dir, 'model_common_init_state.pkl')))
            self.val_dataset = dataset
            print(f'Training Size: {len(self.train_dataset)}; Validation Size: {len(self.val_dataset)}')

            cv_train_transform = transforms.Compose(self.train_transform + [transforms.Normalize(mean=self.train_mean, std=self.train_std)])
            cv_val_transform = transforms.Compose(self.test_transform + [transforms.Normalize(mean=self.train_mean, std=self.train_std)])

            for t in range(k):
                if t != i:
                    k_datasets[t].dataset.transform = cv_train_transform

            self.train_dataset = torch.utils.data.ConcatDataset(
                [k_datasets[t] for t in range(k) if t != i]
            )
            self.val_dataset.dataset.transform = cv_val_transform
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )

            self.train_model(output_suffix=f'_fold{i}')

        val_files = [os.path.join(self.output_dir, f'val_training_loss_fold{i}.csv') for i in range(k)]
        val_dfs = [pd.read_csv(f) for f in val_files]
        val_df = pd.concat(val_dfs)
        val_avg_df = val_df.groupby('epoch').mean()[['loss','acc']].reset_index()
        val_avg_df.to_csv(os.path.join(self.output_dir, f'cv_result{cv_suffix}.csv'), index=False)

    def train_model(self, output_suffix='', save_all=False):
        train_loss_progress = {}
        train_acc_progress = {}
        val_loss_progress = {}
        val_acc_progress = {}
        print(f'Total number of batches: {len(self.train_loader)}')

        best_loss = None
        # Iterate x epochs over the train data
        for epoch in range(self.epochs):
            print('current epoch: ', epoch)
            epoch_loss = []
            epoch_acc = []
            self.model.train()
            epoch_tracker = tqdm(self.train_loader)
            for batch in epoch_tracker:
                epoch_tracker.set_description(
                    f"loss: {np.average(epoch_loss) if len(epoch_loss) > 0 else None}; acc: {np.average(epoch_acc) if len(epoch_acc) > 0 else None}"
                )

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
                epoch_loss.append(loss.item())
                epoch_acc.append(acc)

            train_loss = np.average(epoch_loss)
            train_acc = np.average(epoch_acc)
            train_loss_progress[epoch] = train_loss
            train_acc_progress[epoch] = train_acc
            print(f'epoch training loss: {train_loss}')
            print(f'epoch training acc : {train_acc}')

            if save_all:
                torch.save(self.model.state_dict(),
                           os.path.join(self.output_dir, f'model_epoch{epoch}{output_suffix}.pkl'))

            if self.val_loader is not None:
                self.model.eval()
                val_labels, val_outputs, val_loss, val_report = self.evaluate_model(self.val_loader)
                val_acc = val_report['accuracy']
                val_loss_progress[epoch] = val_loss
                val_acc_progress[epoch] = val_acc
                if best_loss is None:
                    best_loss = val_loss
                    print(f'current epoch has smallest loss value: epoch {epoch}')
                    print('replacing best model file')
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'best_model{output_suffix}.pkl'))
                else:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        print(f'current epoch has smallest loss value: epoch {epoch}')
                        print('replacing best model file')
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(self.output_dir, f'best_model{output_suffix}.pkl')
                        )

                print(f"Epoch {epoch}: train_loss {train_loss}; train_acc {train_acc}; val_loss {val_loss}; val_acc {val_acc}")

            if self.val_loader is None:
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'model_epoch{epoch}{output_suffix}.pkl'))
                if best_loss is None:
                    best_loss = train_loss
                    print(f'current epoch has smallest loss value: epoch {epoch}')
                    print('replacing best model file')
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'best_model{output_suffix}.pkl'))
                else:
                    if train_loss < best_loss:
                        best_loss = np.average(epoch_loss)
                        print(f'current epoch has smallest loss value: epoch {epoch}')
                        print('replacing best model file')
                        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'best_model{output_suffix}.pkl'))

                print(f"Epoch {epoch}: train_loss {train_loss}; train_acc {train_acc}")

        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'final_model{output_suffix}.pkl'))
        training_loss = pd.DataFrame(
            {'epoch': train_loss_progress.keys(), 'loss':train_loss_progress.values(),'acc': train_acc_progress.values()}
        )
        training_loss.to_csv(os.path.join(self.output_dir, f'training_loss{output_suffix}.csv'))

        if val_loss_progress != {}:
            validation_loss = pd.DataFrame(
                {'epoch': val_loss_progress.keys(), 'loss':val_loss_progress.values(),'acc': val_acc_progress.values()}
            )
            validation_loss.to_csv(os.path.join(self.output_dir, f'val_training_loss{output_suffix}.csv'))

    def load_model(self, model_file='best_model.pkl'):
        assert self.model is not None
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, model_file)))

    def evaluate_model(self, input_loader=None):
        assert self.model is not None
        all_labels = np.array([])
        all_outputs = np.array([])
        eval_loss = 0.0

        self.model.eval()
        if input_loader is None:
            input_loader = self.test_loader
        for i, batch in enumerate(input_loader, 0):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            eval_loss += loss.item()
            outputs = outputs.max(1).indices.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            all_labels = np.concatenate((all_labels, labels), axis=None)
            all_outputs = np.concatenate((all_outputs, outputs), axis=None)

        eval_loss = eval_loss/len(input_loader)
        print(classification_report(all_labels, all_outputs))
        report = classification_report(all_labels, all_outputs, output_dict=True)

        return all_labels, all_outputs, eval_loss, report


if __name__ == "__main__":
    from models.CNN import CNNModel

    np.set_printoptions(threshold=sys.maxsize)

    vgg16_model = CNNModel(8)

    train_transform = [
        transforms.RandomResizedCrop(size=(255, 255), scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ]

    dev_classifier = SkinDiseaseClassifier(
        vgg16_model,
        epochs=2,
        batch_size=2,
        output_dir='dev_model_result'
    )
    dev_classifier.create_dataloader(
        train_root_path='dev_images/train',
        test_root_path='dev_images/test',
        train_transform=train_transform,
        seed=57
    )
    # dev_classifier.cross_validate(k=5)
    dev_classifier.train_model()
    dev_classifier.load_model()
    dev_classifier.evaluate_model()