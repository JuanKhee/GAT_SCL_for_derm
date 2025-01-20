import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GATLayerEdgeSoftmax(nn.Module):
    """
    GAT layer with softmax attention distribution (May be prone to numerical errors)
    """

    def __init__(self, d_i, d_o, act=F.leaky_relu, eps=1e-6, dropout=0.3):
        super(GATLayerEdgeSoftmax, self).__init__()
        self.act = act
        self.eps = eps

        self.W_in = nn.Linear(2 * d_i, d_o) # weight parameters for input neighbours
        # self.a = nn.Linear(2 * d_i, 1) # GAT
        self.a = nn.Linear(d_o, 1) # GATV2
        self.W_out = nn.Linear(d_i, d_o)

        self._init_weights()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # print('init W_in', self.W_in.weight)
        # print('init a', self.a.weight)
        # print('init W_out', self.W_out.weight)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.a.weight)
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, x, adj, src, tgt, Msrc, Mtgt, return_attention=False):
        """
        features -> N,i node features
        adj -> N,N adjacency matrix
        src -> E,i source index for edges
        tgt -> E,i target index for edges
        Msrc -> N,E adjacency matrix from source nodes to edges
        Mtgt -> N,E adjacency matrix from target nodes to edges
        # unused method parameters are to ensure consistency from model input to layer inputs
        """
        hsrc = x[src]  # extract source of each edge
        htgt = x[tgt]  # extract target of each edge
        h = torch.cat([hsrc, htgt], dim=1)  # concatenate features of source and target
        wh = self.W_in(h) # Apply W weight matrix onto features
        sig_wh = self.act(wh)  # Apply non-linear activation onto weighted features
        sig_wh = self.dropout1(sig_wh)
        e = self.a(sig_wh)  # GATV2 - apply a weight matrix onto activated weighted features to obtain raw attention
        assert not torch.isnan(e).any()

        exp_e = torch.exp(e) # get exp of each raw attention
        assert not torch.isnan(exp_e).any()

        exp_e_sum = torch.mm(Mtgt, exp_e) + self.eps  # get sum of attention for all target nodes of each source
        exp_e_sum = exp_e_sum[tgt] # reindex sum values for parallel computation
        assert not torch.isnan(exp_e_sum).any()

        # alpha = torch.mm(Mtgt, y * a_exp) / a_sum  # GAT
        alpha = exp_e / exp_e_sum  # GATV2 - softmax raw attention to obtain final attention values
        assert not torch.isnan(alpha).any()

        w2_hout = self.W_out(htgt)
        assert not torch.isnan(w2_hout).any()

        alpha_w2_hout = alpha * w2_hout
        assert not torch.isnan(alpha_w2_hout).any()

        h_new_raw = torch.mm(Mtgt, alpha_w2_hout)
        assert not torch.isnan(h_new_raw).any()

        h_new_act = self.act(h_new_raw)
        assert not torch.isnan(h_new_act).any()
        h_new_act = self.dropout1(h_new_act)

        if return_attention:
            return h_new_act, alpha

        return h_new_act


class GATLayerMultiHead(nn.Module):

    def __init__(self, d_in, d_out, num_heads, dropout=0.3):
        super(GATLayerMultiHead, self).__init__()

        self.GAT_heads = nn.ModuleList(
            [
                GATLayerEdgeSoftmax(d_in, d_out, dropout=dropout)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x, adj, src, tgt, Msrc, Mtgt):
        return torch.cat([l(x, adj, src, tgt, Msrc, Mtgt) for l in self.GAT_heads], dim=1)


class GAT_image(nn.Module):

    def __init__(self, num_features, num_classes, num_heads=[2, 2, 2], layer_sizes=[32,64,64], dropouts=[0.3,0.3,0.3]):
        super(GAT_image, self).__init__()

        self.layer_heads = [1] + num_heads
        # Computes attention coefficients and outputs transformed features
        self.GAT_layer_sizes = [num_features] + list(layer_sizes)

        # Applies Multihead framework for attention computation
        self.MLP_layer_sizes = [self.layer_heads[-1] * self.GAT_layer_sizes[-1], 32, num_classes]
        self.MLP_acts = [F.leaky_relu, lambda x: x]

        self.dropouts = dropouts

        self.GAT_layers = nn.ModuleList(
            [
                GATLayerMultiHead(d_in * heads_in, d_out, heads_out, dropout=dropout)
                for d_in, d_out, heads_in, heads_out, dropout in zip(
                    self.GAT_layer_sizes[:-1],
                    self.GAT_layer_sizes[1:],
                    self.layer_heads[:-1],
                    self.layer_heads[1:],
                    self.dropouts
                )
            ]
        )
        self.MLP_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(self.MLP_layer_sizes[:-1], self.MLP_layer_sizes[1:])
            ]
        )

    def forward(self, x, adj, src, tgt, Msrc, Mtgt, Mgraph):
        for l in self.GAT_layers:
            x = l(x, adj, src, tgt, Msrc, Mtgt)
        x = torch.mm(Mgraph.t(), x)
        for layer, act in zip(self.MLP_layers, self.MLP_acts):
            x = act(layer(x))
        return x


if __name__ == "__main__":
    g = GATLayerEdgeSoftmax(3, 10)
    x0 = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
    adj = torch.tensor([[0, 1], [1, 0]])
    src = [0, 1]
    tgt = [1, 0]
    Msrc = torch.tensor([[1.,0.], [0.,1.]])
    Mtgt = torch.tensor([[0.,1.], [1.,0.]])
    Mgraph = [[1], [1]]

    x1 = g(x0, adj, src, tgt, Msrc, Mtgt)
    print(x1)