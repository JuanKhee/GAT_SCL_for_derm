import torch
import torchvision.transforms as transforms

from models.CNN import CNNModel
from models.GAT_superpixel import GAT_image
from utils.XAI.GAT_viz import visualize_attention
from utils.graph_utils import batch_graphs, ImageDatasetWithFile


# Example usage
cnn_output = 32
gat_features = 8
gat_output = 16
num_heads = [2, 2, 4]
layer_sizes = [16, 32, 32]
dropouts = [0, 0.3, 0]
metadata_features = 8
num_classes = 8

# Instantiate and load your model
cnn_model = CNNModel(32, 'resnet152')
gat_model = GAT_image(gat_features, gat_output, num_heads, layer_sizes, dropouts)
mlp_model = torch.nn.Linear(cnn_output+gat_output+metadata_features, num_classes)

cnn_model.load_state_dict(
    torch.load(
        r"C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\final_results\cnn_model_final.pkl",
        map_location=torch.device('cpu')
    )
)
gat_model.load_state_dict(
    torch.load(
        r"C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\final_results\gat_model_final.pkl",
        map_location=torch.device('cpu')
    )
)
mlp_model.load_state_dict(
    torch.load(
        r"C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\final_results\mlp_model_final.pkl",
        map_location=torch.device('cpu')
    )
)
cnn_model.eval()
gat_model.eval()
mlp_model.eval()

test_dataset = ImageDatasetWithFile(
    torchvision.datasets.ImageFolder(
        root=r"dev_images\test",
        transform=transforms.Compose(test_transform + [transforms.Normalize(mean=self.train_mean, std=self.train_mean)])
    ),
    self.test_metadata,
    output_dir=test_graph_dir,
    output_file_suffix=output_file_suffix
)
self.test_loader = torch.utils.data.DataLoader(
    self.test_dataset,
    batch_size=self.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=self.num_workers,
    pin_memory=self.pin_memory
)

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
# # Define inputs for the model
# x = torch.rand((100, gat_features))  # 100 nodes with 8 features each
# adj = torch.randint(0, 2, (100, 100))  # Adjacency matrix
# src, tgt = torch.where(adj)  # Edge source and target indices
# Msrc = torch.eye(len(src))  # Edge-to-source mapping
# Mtgt = torch.eye(len(tgt))  # Edge-to-target mapping
# Mgraph = torch.eye(100)  # Global adjacency matrix
#
# # Visualize attention for a specific layer and head
# visualize_attention(model, x, adj, src, tgt, Msrc, Mtgt, Mgraph, layer_idx=1, head_idx=0)