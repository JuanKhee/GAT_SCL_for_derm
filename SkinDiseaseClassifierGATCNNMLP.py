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
from utils.graph_utils import batch_graphs, ImageDatasetWithFile
from tqdm import tqdm
from utils.data_utils import compute_mean_std
from utils.graph_utils import ImageGraphDualTransform


class SkinDiseaseClassifier():
    def __init__(
            self,
            cnn_model,
            gat_model,
            mlp_model,
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

        self.cnn_model = cnn_model
        self.gat_model = gat_model
        self.mlp_model = mlp_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")

        self.cnn_model.to(self.device)
        self.gat_model.to(self.device)
        self.mlp_model.to(self.device)

        self.criterion = criterion
        self.optimizer = optimizer(
            list(self.cnn_model.parameters()) + list(self.gat_model.parameters()) + list(self.mlp_model.parameters()),
            lr=learning_rate
        )

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
            train_metadata_path,
            test_metadata_path,
            seed=0,
            train_transform=None,
            test_transform=None,
            collate_fn=None,
            train_graph_dir="Training_Graphs_50_nodes",
            test_graph_dir="Test_Graphs_50_nodes",
            output_file_suffix='.npy'
    ):
        self.train_root_path = train_root_path
        self.test_root_path = test_root_path
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.collate_fn = collate_fn

        self.train_metadata_path = train_metadata_path
        self.test_metadata_path = test_metadata_path

        self.train_metadata = pd.read_csv(self.train_metadata_path)
        self.test_metadata = pd.read_csv(self.test_metadata_path)

        torch.manual_seed(seed)
        self.train_dataset = ImageDatasetWithFile(
            torchvision.datasets.ImageFolder(
                root=train_root_path,
                # transform=ImageGraphDualTransform(
                #     img_transform=transforms.Compose(self.train_transform[0]),
                #     graph_transform=transforms.Compose(self.train_transform[1])
                # ),
                transform=transforms.Compose(self.train_transform + [transforms.Normalize(mean=self.train_mean, std=self.train_mean)])
            ),
            self.train_metadata,
            output_dir=train_graph_dir,
            output_file_suffix=output_file_suffix
        )
        self.test_dataset = ImageDatasetWithFile(
            torchvision.datasets.ImageFolder(
                root=test_root_path,
                # transform=ImageGraphDualTransform(
                #     img_transform=transforms.Compose(self.test_transform[0]),
                #     graph_transform=transforms.Compose(self.test_transform[1])
                # )
                transform=transforms.Compose(self.test_transform + [transforms.Normalize(mean=self.train_mean, std=self.train_mean)])
            ),
            self.test_metadata,
            output_dir=test_graph_dir,
            output_file_suffix=output_file_suffix
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        self.val_dataset = None
        self.val_loader = None

    def cross_validate(self, k=2, seed=0, cv_suffix=''):

        assert 100 % k == 0
        torch.manual_seed(seed)
        k_datasets = torch.utils.data.random_split(self.train_dataset, [1/k]*k)
        torch.save(self.cnn_model.state_dict(), os.path.join(self.output_dir, f'cnn_model_common_init_state.pkl'))
        torch.save(self.gat_model.state_dict(), os.path.join(self.output_dir, f'gat_model_common_init_state.pkl'))
        torch.save(self.mlp_model.state_dict(), os.path.join(self.output_dir, f'mlp_model_common_init_state.pkl'))
        for i, dataset in enumerate(k_datasets):
            print(f'fold {i+1}/{k}')
            self.cnn_model.load_state_dict(torch.load(os.path.join(self.output_dir, 'cnn_model_common_init_state.pkl')))
            self.gat_model.load_state_dict(torch.load(os.path.join(self.output_dir, 'gat_model_common_init_state.pkl')))
            self.mlp_model.load_state_dict(torch.load(os.path.join(self.output_dir, 'mlp_model_common_init_state.pkl')))
            self.val_dataset = dataset
            print(f'Training Size: {len(self.train_dataset)}; Validation Size: {len(self.val_dataset)}', flush=True)

            cv_train_transform = transforms.Compose(
                self.train_transform + [transforms.Normalize(mean=self.train_mean, std=self.train_std)])
            cv_val_transform = transforms.Compose(
                self.test_transform + [transforms.Normalize(mean=self.train_mean, std=self.train_std)])

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
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
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
            self.cnn_model.train()
            self.gat_model.train()
            self.mlp_model.train()
            epoch_tracker = tqdm(self.train_loader)
            for batch in epoch_tracker:
                epoch_tracker.set_description(
                    f"loss: {np.average(epoch_loss) if len(epoch_loss) > 0 else None}; acc: {np.average(epoch_acc) if len(epoch_acc) > 0 else None}"
                )

                inputs, metadata_input, labels = batch
                cnn_inputs = torch.tensor([inp[0].cpu().numpy() for inp in inputs])
                gat_inputs = [inp[1] for inp in inputs]
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

                metadata_input = metadata_input.to(self.device)

                self.optimizer.zero_grad()
                cnn_outputs = self.cnn_model(cnn_inputs)
                # cnn_outputs = torch.nn.Sigmoid()(cnn_outputs)
                gat_outputs = self.gat_model(h, adj, src, tgt, Msrc, Mtgt, Mgraph)
                # gat_outputs = torch.nn.Sigmoid()(gat_outputs)

                deep_block_output = cnn_outputs + gat_outputs
                mlp_input = torch.cat([deep_block_output, metadata_input], dim=1)
                outputs = self.mlp_model(mlp_input)

                # outputs = torch.nn.Softmax(dim=1)(outputs)

                # CEloss calls softmax implicitly
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
                torch.save(
                    self.cnn_model.state_dict(),
                    os.path.join(self.output_dir, f'cnn_model_epoch{epoch}{output_suffix}.pkl')
                )
                torch.save(
                    self.gat_model.state_dict(),
                    os.path.join(self.output_dir, f'gat_model_epoch{epoch}{output_suffix}.pkl')
                )
                torch.save(
                    self.mlp_model.state_dict(),
                    os.path.join(self.output_dir, f'mlp_model_epoch{epoch}{output_suffix}.pkl')
                )

            if self.val_loader is not None:
                self.cnn_model.eval()
                self.gat_model.eval()
                self.mlp_model.eval()
                val_labels, val_outputs, val_loss, val_report = self.evaluate_model(self.val_loader)
                val_acc = val_report['accuracy']
                val_loss_progress[epoch] = val_loss
                val_acc_progress[epoch] = val_acc
                if best_loss is None:
                    best_loss = val_loss
                    print(f'current epoch has smallest loss value: epoch {epoch}')
                    print('replacing best model file')
                    torch.save(self.cnn_model.state_dict(),
                               os.path.join(self.output_dir, f'best_cnn_model{output_suffix}.pkl'))
                    torch.save(self.gat_model.state_dict(),
                               os.path.join(self.output_dir, f'best_gat_model{output_suffix}.pkl'))
                    torch.save(self.mlp_model.state_dict(),
                               os.path.join(self.output_dir, f'best_mlp_model{output_suffix}.pkl'))
                else:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        print(f'current epoch has smallest loss value: epoch {epoch}')
                        print('replacing best model file')
                        torch.save(self.cnn_model.state_dict(),
                                   os.path.join(self.output_dir, f'best_cnn_model{output_suffix}.pkl'))
                        torch.save(self.gat_model.state_dict(),
                                   os.path.join(self.output_dir, f'best_gat_model{output_suffix}.pkl'))
                        torch.save(self.mlp_model.state_dict(),
                                   os.path.join(self.output_dir, f'best_mlp_model{output_suffix}.pkl'))

                print(
                    f"Epoch {epoch}: train_loss {train_loss}; train_acc {train_acc}; val_loss {val_loss}; val_acc {val_acc}")

            if self.val_loader is None:
                if best_loss is None:
                    best_loss = train_loss
                    print(f'current epoch has smallest loss value: epoch {epoch}')
                    print('replacing best model file')
                    torch.save(self.cnn_model.state_dict(),
                               os.path.join(self.output_dir, f'best_cnn_model{output_suffix}.pkl'))
                    torch.save(self.gat_model.state_dict(),
                               os.path.join(self.output_dir, f'best_gat_model{output_suffix}.pkl'))
                    torch.save(self.mlp_model.state_dict(),
                               os.path.join(self.output_dir, f'best_mlp_model{output_suffix}.pkl'))
                else:
                    if train_loss < best_loss:
                        best_loss = np.average(epoch_loss)
                        print(f'current epoch has smallest loss value: epoch {epoch}')
                        print('replacing best model file')
                        torch.save(self.cnn_model.state_dict(),
                                   os.path.join(self.output_dir, f'best_cnn_model{output_suffix}.pkl'))
                        torch.save(self.gat_model.state_dict(),
                                   os.path.join(self.output_dir, f'best_gat_model{output_suffix}.pkl'))
                        torch.save(self.mlp_model.state_dict(),
                                   os.path.join(self.output_dir, f'best_mlp_model{output_suffix}.pkl'))

                print(f"Epoch {epoch}: train_loss {train_loss}; train_acc {train_acc}")

        torch.save(self.cnn_model.state_dict(),
                   os.path.join(self.output_dir, f'final_cnn_model{output_suffix}.pkl'))
        torch.save(self.gat_model.state_dict(),
                   os.path.join(self.output_dir, f'final_gat_model{output_suffix}.pkl'))
        torch.save(self.mlp_model.state_dict(),
                   os.path.join(self.output_dir, f'final_mlp_model{output_suffix}.pkl'))

        training_loss = pd.DataFrame(
            {'epoch': train_loss_progress.keys(), 'loss': train_loss_progress.values(),
             'acc': train_acc_progress.values()}
        )
        training_loss.to_csv(os.path.join(self.output_dir, f'training_loss{output_suffix}.csv'))

        if val_loss_progress != {}:
            validation_loss = pd.DataFrame(
                {'epoch': val_loss_progress.keys(), 'loss': val_loss_progress.values(),
                 'acc': val_acc_progress.values()}
            )
            validation_loss.to_csv(os.path.join(self.output_dir, f'val_training_loss{output_suffix}.csv'))

    def load_model(self):
        assert self.cnn_model is not None
        assert self.gat_model is not None
        assert self.mlp_model is not None
        self.cnn_model.load_state_dict(torch.load(os.path.join(self.output_dir, 'best_cnn_model.pkl')))
        self.gat_model.load_state_dict(torch.load(os.path.join(self.output_dir, 'best_gat_model.pkl')))
        self.mlp_model.load_state_dict(torch.load(os.path.join(self.output_dir, 'best_mlp_model.pkl')))

    def evaluate_model(self, input_loader=None):
        assert self.cnn_model is not None
        assert self.gat_model is not None
        assert self.mlp_model is not None
        print(f'Test size: {len(self.test_dataset)}')
        all_labels = np.array([])
        all_outputs = np.array([])
        eval_loss = 0.0

        self.cnn_model.eval()
        self.gat_model.eval()
        self.mlp_model.eval()
        if input_loader is None:
            input_loader = self.test_loader
        for i, batch in enumerate(input_loader, 0):
            inputs, metadata_input, labels = batch
            cnn_inputs = torch.tensor([inp[0].cpu().numpy() for inp in inputs])
            gat_inputs = [inp[1] for inp in inputs]
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

            metadata_input = metadata_input.to(self.device)

            self.optimizer.zero_grad()
            cnn_outputs = self.cnn_model(cnn_inputs)
            cnn_outputs = torch.nn.Sigmoid()(cnn_outputs)
            gat_outputs = self.gat_model(h, adj, src, tgt, Msrc, Mtgt, Mgraph)
            gat_outputs = torch.nn.Sigmoid()(gat_outputs)

            deep_block_output = cnn_outputs + gat_outputs
            mlp_input = torch.cat([deep_block_output, metadata_input], dim=1)
            outputs = self.mlp_model(mlp_input)
            # outputs = torch.nn.Softmax(dim=1)(outputs)

            loss = self.criterion(outputs, labels)
            eval_loss += loss.item()
            outputs = outputs.max(1).indices.detach().cpu().numpy()
            all_labels = np.concatenate((all_labels, labels), axis=None)
            all_outputs = np.concatenate((all_outputs, outputs), axis=None)

        eval_loss = eval_loss/len(input_loader)
        print(classification_report(all_labels, all_outputs))
        report = classification_report(all_labels, all_outputs, output_dict=True)

        return all_labels, all_outputs, eval_loss, report


if __name__ == "__main__":
    from models.CNN import CNNModel
    from models.GAT_superpixel import GAT_image
    from utils.graph_utils import ImgToGraphTransform, graph_metadata_collate, GRAPH_FEATURES, METADATA_FEATURES
    np.set_printoptions(threshold=sys.maxsize)

    DEEP_BLOCK_OUTPUT = 16
    NUM_CLASSES = 8

    CNN_model = CNNModel(DEEP_BLOCK_OUTPUT)
    GAT_model = GAT_image(GRAPH_FEATURES,DEEP_BLOCK_OUTPUT, num_heads=[2, 2, 2], layer_sizes=[32,64,64])
    MLP_model = torch.nn.Linear(DEEP_BLOCK_OUTPUT+METADATA_FEATURES, NUM_CLASSES)
    nn.init.xavier_uniform_(MLP_model.weight)
    
    dev_classifier = SkinDiseaseClassifier(
        cnn_model=CNN_model,
        gat_model=GAT_model,
        mlp_model=MLP_model,
        epochs=2,
        batch_size=8,
        output_dir='dev_model_result_gatcnnmlp_metadata'
    )
    dev_classifier.create_dataloader(
        train_root_path=r'dev_images\train',
        test_root_path=r'dev_images\test',
        train_metadata_path=r'metadata\ISIC_2019_Training_Metadata.csv',
        test_metadata_path=r'metadata\ISIC_2019_Test_Metadata.csv',
        train_transform=[
            # [
                transforms.RandomResizedCrop(size=(255, 255), scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            # ],
            # [
            #     transforms.ToTensor()
            # ]
        ],
        test_transform=[
            # [
                transforms.Resize(size=(255, 255)),
                transforms.ToTensor(),
            # ],
            # [
            #     transforms.ToTensor()
            # ]
        ]
        ,
        collate_fn=graph_metadata_collate,
        seed=57,
        train_graph_dir="Training_Graph_60_nodes",
        test_graph_dir="Test_Graph_60_nodes",
        output_file_suffix='.npy'
    )
    dev_classifier.cross_validate(k=2)
    # print(dev_classifier.train_dataset[0])
    dev_classifier.train_model()
    dev_classifier.load_model()
    dev_classifier.evaluate_model()