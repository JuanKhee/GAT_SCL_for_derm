import sys
import numpy as np

if __name__ == "__main__":
    sys.path.append(r"C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm")
    from SkinDiseaseClassifierGATCNNMLP_MTSCL import SkinDiseaseClassifier
    from models.CNN import CNNModel
    from models.GAT_superpixel import GAT_image
    from utils.graph_utils import graph_metadata_collate, GRAPH_FEATURES, \
        METADATA_FEATURES, batch_graphs
    import torchvision.transforms as transforms
    import torch.optim as optim
    from torch import nn
    import torch
    from tqdm import tqdm
    from sklearn.metrics import classification_report

    deep_output, num_heads, gat_layer_size = (32, [2, 2, 4], [16, 32, 32])
    gat_output = 16
    dropouts = [0, 0.3, 0]
    NUM_CLASSES = 8
    torch.manual_seed(17)

    CNN_model = CNNModel(deep_output, 'resnet152')
    GAT_model = GAT_image(GRAPH_FEATURES, gat_output, num_heads=num_heads, layer_sizes=gat_layer_size, dropouts=dropouts)
    MLP_model = torch.nn.Linear(deep_output + gat_output + METADATA_FEATURES, NUM_CLASSES)
    nn.init.xavier_uniform_(MLP_model.weight)

    classifier = SkinDiseaseClassifier(
        cnn_model=CNN_model,
        gat_model=GAT_model,
        mlp_model=MLP_model,
        epochs=2,
        batch_size=1,
        output_dir='prd_model_result_gatcnnmlp_metadata'
    )
    classifier.create_dataloader(
        train_root_path=r'C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\dev_images\test',
        test_root_path=r'C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\dev_images\test',
        train_metadata_path=r'C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\metadata\ISIC_2019_Test_Metadata.csv',
        test_metadata_path=r'C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\metadata\ISIC_2019_Test_Metadata.csv',
        train_transform=[
            transforms.RandomResizedCrop(size=(255, 255), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ],
        test_transform=[
            transforms.Resize(size=(255, 255)),
            transforms.CenterCrop(size=(224,224)),
            transforms.ToTensor(),
        ]
        ,
        collate_fn=graph_metadata_collate,
        seed=57,
        train_graph_dir="test_Graph_60_nodes",
        test_graph_dir="test_Graph_60_nodes",
        output_file_suffix='.npy'
    )

    classifier.cnn_model.load_state_dict(
        torch.load(r"C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\final_results\cnn_model_final_cpu.pkl"
    ))
    classifier.gat_model.load_state_dict(
        torch.load(r"C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\final_results\gat_model_final_cpu.pkl"
    ))
    classifier.mlp_model.load_state_dict(
        torch.load(r"C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\final_results\mlp_model_final_cpu.pkl"
    ))

    all_labels = np.array([])
    all_raw_outputs = torch.tensor([])
    all_outputs = np.array([])
    eval_loss = 0.0
    classifier.cnn_model.eval()
    classifier.gat_model.eval()
    classifier.mlp_model.eval()

    for i, batch in tqdm(enumerate(classifier.test_loader, 0)):
        with torch.no_grad():
            inputs, metadata_input, labels = batch
            cnn_inputs = torch.tensor(np.array([inp[0].cpu().numpy() for inp in inputs]))
            gat_inputs = [inp[1] for inp in inputs]
            gat_batch = (gat_inputs, labels)
            h, adj, src, tgt, Msrc, Mtgt, Mgraph, gat_labels = batch_graphs(gat_batch)
            h, adj, src, tgt, Msrc, Mtgt, Mgraph = map(
                torch.from_numpy,
                (h, adj, src, tgt, Msrc, Mtgt, Mgraph)
            )
            h = h.to(classifier.device)
            adj = adj.to(classifier.device)
            src = src.to(classifier.device)
            tgt = tgt.to(classifier.device)
            Msrc = Msrc.to(classifier.device)
            Mtgt = Mtgt.to(classifier.device)
            Mgraph = Mgraph.to(classifier.device)
            gat_labels = gat_labels.to(classifier.device)

            cnn_inputs = cnn_inputs.to(classifier.device)
            labels = labels.to(classifier.device)

            metadata_input = metadata_input.to(classifier.device)

            classifier.optimizer.zero_grad()
            cnn_outputs = classifier.cnn_model(cnn_inputs)
            gat_outputs = classifier.gat_model(h, adj, src, tgt, Msrc, Mtgt, Mgraph)

            deep_block_output = torch.cat([cnn_outputs, gat_outputs], dim=1)
            mlp_input = torch.cat([deep_block_output, metadata_input], dim=1)
            outputs = classifier.mlp_model(mlp_input)
            all_raw_outputs = torch.cat([all_raw_outputs, outputs], dim=0)
            loss = classifier.criterion(outputs, labels)
            eval_loss += loss.item()
            outputs = outputs.max(1).indices.detach().cpu().numpy()
            all_labels = np.concatenate((all_labels, labels.cpu().numpy()), axis=None)
            all_outputs = np.concatenate((all_outputs, outputs), axis=None)


            print(outputs, labels.detach().cpu().numpy()[0], classifier.test_dataset.dataset.imgs[i][0].split('\\')[-1])

    eval_loss = eval_loss / len(classifier.test_loader)
    print(classification_report(all_labels, all_outputs))
    report = classification_report(all_labels, all_outputs, output_dict=True)