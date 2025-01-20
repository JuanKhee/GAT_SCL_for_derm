import matplotlib.pyplot as plt
import networkx as nx
import torch


def visualize_attention(model, x, adj, src, tgt, Msrc, Mtgt, Mgraph, layer_idx=0, head_idx=0):
    """
    Visualize attention for a specific layer and head in the GAT model.

    Parameters:
        model: The trained GAT model.
        x: Input features (nodes of the graph).
        adj: Adjacency matrix of the graph.
        src: Source indices for edges.
        tgt: Target indices for edges.
        Msrc, Mtgt: Matrices mapping edges to nodes.
        Mgraph: Global graph adjacency matrix for the dataset.
        layer_idx: Index of the GAT layer to visualize.
        head_idx: Index of the attention head in the chosen layer.
    """
    # Forward pass to extract attention weights
    attentions = []
    current_x = x
    for i, layer in enumerate(model.GAT_layers):
        current_attentions = []
        for j, head in enumerate(layer.GAT_heads):
            output, attention = head(current_x, adj, src, tgt, Msrc, Mtgt, return_attention=True)
            if i == layer_idx and j == head_idx:
                attentions = attention
            current_attentions.append(attention)
        current_x = torch.cat([output for output, _ in current_attentions], dim=1)

    # Build graph for visualization
    G = nx.from_numpy_matrix(adj.cpu().numpy())
    pos = nx.spring_layout(G)  # Position nodes using a spring layout

    # Normalize attention values for visualization
    attentions = attentions.detach().cpu().numpy().squeeze()
    edge_colors = [attentions[k] for k in range(len(attentions))]

    # Plot the graph
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color='skyblue',
        edge_color=edge_colors,
        edge_cmap=plt.cm.Blues,
        node_size=500,
        font_size=10
    )
    plt.title(f"Attention Visualization: Layer {layer_idx}, Head {head_idx}")
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues), label="Attention Weight")
    plt.show()

