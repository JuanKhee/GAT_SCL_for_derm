import torch
import numpy as np
import copy

nodes = torch.tensor([
    [0.74,0.55,0.54,0.25,0.25],
    [0.76,0.59,0.59,0.76,0.24],
    [0.74,0.53,0.53,0.73,0.74],
    [0.74,0.54,0.54,0.24,0.75]
])

adjacencies = np.array([
    [0, 0],
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 0],
    [1, 1],
    [1, 2],
    [2, 0],
    [2, 1],
    [2, 2],
    [2, 3],
    [3, 0],
    [3, 2],
    [3, 3]
])

layer_sizes = [10,2,2,2]
layer_inputs = layer_sizes[:-1]
layer_outputs = layer_sizes[1:]
layer_dim  = [i for i in zip(layer_inputs, layer_outputs)]

act = torch.nn.LeakyReLU()
loss = torch.nn.CrossEntropyLoss()

label = torch.tensor([1])

graph_nodes = {i:None for i in range(len(layer_dim) + 1)}
graph_nodes[0] = copy.deepcopy(nodes)
W_in = {}
a = {}
W_out = {}
feat_size = 5

for i, dim in enumerate(layer_dim):
    print(dim)
    W_in[i] = torch.nn.Linear(*dim)
    # a[i] = torch.nn.Linear(dim[1],dim[1])
    a[i] = torch.nn.Linear(dim[1],1)
    W_out[i] = torch.nn.Linear(feat_size,dim[1])
    with torch.no_grad():
        W_in[i].weight = torch.nn.Parameter(torch.ones(dim).T)
        W_in[i].bias = torch.nn.Parameter(torch.ones(dim[1]))
        a[i].weight = torch.nn.Parameter(torch.ones(dim[1],1).T)
        a[i].bias = torch.nn.Parameter(torch.ones(1))
        W_out[i].weight = torch.nn.Parameter(torch.ones(feat_size,dim[1]).T)
        W_out[i].bias = torch.nn.Parameter(torch.ones(dim[1]))


NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64

for layer in range(len(layer_dim)):
    G = 1
    N = graph_nodes[layer].shape[0]
    M = adjacencies.shape[0]

    h = graph_nodes[layer]
    adj = np.zeros([N, N])
    src = np.zeros([M])
    tgt = np.zeros([M])
    Msrc = np.zeros([N, M])
    Mtgt = np.zeros([N, M])
    Mgraph = np.zeros([N, G])

    adj.astype(NP_TORCH_FLOAT_DTYPE)
    src.astype(NP_TORCH_LONG_DTYPE)
    tgt.astype(NP_TORCH_LONG_DTYPE)
    Msrc.astype(NP_TORCH_FLOAT_DTYPE)
    Mtgt.astype(NP_TORCH_FLOAT_DTYPE)
    Mgraph.astype(NP_TORCH_FLOAT_DTYPE)

    n_acc = 0
    m_acc = 0

    for e, (s, t) in enumerate(adjacencies):
        adj[n_acc + s, n_acc + t] = 1
        adj[n_acc + t, n_acc + s] = 1

        src[m_acc + e] = n_acc + s
        tgt[m_acc + e] = n_acc + t

        Msrc[n_acc + s, m_acc + e] = 1
        Mtgt[n_acc + t, m_acc + e] = 1

    Mgraph[n_acc:n_acc + N, 0] = 1

    print('h',np.round(h,2))
    print('adj', adj)
    print('src', src)
    print('tgt', tgt)
    print('Msrc', Msrc)
    print('Mtgt', Mtgt)
    print('Mgraph', Mgraph)

    print('cur_W', W_in[layer])
    print('cur_graph', graph_nodes[layer])

    print(h[src])
    print(torch.cat([h[src], h[tgt]], dim=1))

    print(W_in[layer])
    wh = W_in[layer](torch.cat([h[src], h[tgt]], dim=1))
    print(f'wh', wh)

    sig_wh = act(wh)
    print(f'sig_wh', sig_wh)

    e = a[layer](sig_wh)
    print(f'e', e)

    exp_e = torch.exp(e)
    print(f'exp_e', exp_e)

    sum_exp_e = torch.mm(torch.from_numpy(Mtgt), exp_e.type(torch.DoubleTensor))
    print(Mtgt)
    print(f'sum_exp_e', sum_exp_e)
    sum_exp_e = sum_exp_e[tgt]
    print(f'sum_exp_e', sum_exp_e)

    alpha = exp_e/sum_exp_e
    print('alpha', alpha)

    w2_hout = W_out[layer](h[tgt])
    print('w2_hout', w2_hout)

    alpha_w2_hout = alpha * w2_hout.type(torch.DoubleTensor)
    print('alpha_w2_hout', alpha_w2_hout)

    h_new_raw = torch.mm(torch.from_numpy(Mtgt), alpha_w2_hout)
    print('h_new_raw', h_new_raw)

    h_new_act = act(h_new_raw)
    print('h_new_act', h_new_act)
    break

print()