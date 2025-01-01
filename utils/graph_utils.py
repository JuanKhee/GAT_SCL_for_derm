from tqdm import tqdm
import fire

import time
import pickle
import multiprocessing

import numpy as np
import scipy as sp
from skimage.segmentation import slic, mark_boundaries
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64

NUM_FEATURES = 5
NUM_CLASSES = 10


# General utils

def save_model(fname, model):
    torch.save(model.state_dict(), "{fname}.pt".format(fname=fname))


def load_model(fname, model):
    model.load_state_dict(torch.load("{fname}.pt".format(fname=fname)))


def to_cuda(x):
    return x.cuda()


# Superpixel utils

def plot_image(image, desired_nodes=75, save_in=None):
    # show the output of SLIC
    fig = plt.figure("Image")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)  # , cmap="gray")
    plt.axis("off")

    # show the plots
    if save_in is None:
        plt.show()
    else:
        plt.savefig(save_in, bbox_inches="tight")
    plt.close()


def plot_graph_from_image(PIL_image, desired_nodes=75, save_in=None):
    # segments = slic(image, slic_zero=True)
    image = np.asarray(PIL_image)
    segments = -1 + slic(image, n_segments=desired_nodes, slic_zero=True)
    np.savetxt("segments.txt", segments, fmt='%i')

    # show the output of SLIC
    fig = plt.figure("Superpixels")
    ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(mark_boundaries(image, segments), cmap="gray")
    ax.imshow(mark_boundaries(image, segments))  # , cmap="gray")
    plt.axis("off")

    asegments = np.array(segments)

    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments

    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])

    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    plt.scatter(centers[:, 1], centers[:, 0], c='r')

    for i in range(bneighbors.shape[1]):
        if bneighbors[0, i] != bneighbors[1, i]:
            y0, x0 = centers[bneighbors[0, i]]
            y1, x1 = centers[bneighbors[1, i]]

            l = Line2D([x0, x1], [y0, y1], c="r", alpha=0.5)
            ax.add_line(l)

    # show the plots
    if save_in is None:
        plt.show()
    else:
        plt.savefig(save_in, bbox_inches="tight")
    plt.close()
    #
    # return centers

# GAT utils

def get_graph_from_image(PIL_image, desired_nodes=75):
    # load the image and convert it to a floating point data type
    image = np.asarray(PIL_image)
    # print(image.shape)
    # exit()
    segments = -1 + slic(image, n_segments=desired_nodes, slic_zero=True)

    num_nodes = np.max(segments)
    nodes = {
        node: {
            "rgb_list": [],
            "pos_list": []
        } for node in range(num_nodes + 1)
    }

    height = image.shape[0]
    width = image.shape[1]
    for y in range(height):
        for x in range(width):
            node = segments[y, x]
            rgb = image[y, x, :]
            pos = np.array([float(x) / width, float(y) / height])
            nodes[node]["rgb_list"].append(rgb)
            nodes[node]["pos_list"].append(pos)
        # end for
    # end for

    G = nx.Graph()

    for node in nodes:
        nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        # rgb
        rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
        # rgb_std = np.std(nodes[node]["rgb_list"], axis=0)
        # rgb_gram = np.matmul( nodes[node]["rgb_list"].T, nodes[node]["rgb_list"] ) / nodes[node]["rgb_list"].shape[0]
        # Pos
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
        # pos_std = np.std(nodes[node]["pos_list"], axis=0)
        # pos_gram = np.matmul( nodes[node]["pos_list"].T, nodes[node]["pos_list"] ) / nodes[node]["pos_list"].shape[0]
        # Debug
        #
        # print(node, rgb_mean, pos_mean)

        features = np.concatenate(
            [
                np.reshape(rgb_mean, -1),
                # np.reshape(rgb_std, -1),
                # np.reshape(rgb_gram, -1),
                np.reshape(pos_mean, -1),
                # np.reshape(pos_std, -1),
                # np.reshape(pos_gram, -1)
            ]
        )
        G.add_node(node, features=list(features))
    # end

    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])

    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    # Adjacency loops
    for i in range(bneighbors.shape[1]):
        if bneighbors[0, i] != bneighbors[1, i]:
            G.add_edge(bneighbors[0, i], bneighbors[1, i])

    # Self loops
    for node in nodes:
        G.add_edge(node, node)

    n = len(G.nodes)
    m = len(G.edges)
    h = np.zeros([n, NUM_FEATURES]).astype(NP_TORCH_FLOAT_DTYPE)
    edges = np.zeros([2 * m, 2]).astype(NP_TORCH_LONG_DTYPE)
    for e, (s, t) in enumerate(G.edges):
        edges[e, 0] = s
        edges[e, 1] = t

        edges[m + e, 0] = t
        edges[m + e, 1] = s
    # end for
    for i in G.nodes:
        h[i, :] = G.nodes[i]["features"]
    # end for
    del G
    return h, edges


def batch_graphs(batch, two_crop=False):
    # batch ~ [((h,edges),int)]
    gs = batch[0]
    if two_crop:
        print('gs', len(gs))
        ori_feat = [two_crop_g[0] for two_crop_g in gs]
        dual_feat = [two_crop_g[1] for two_crop_g in gs]
        gs = ori_feat + dual_feat
        print('gs',len(gs))
    labels = batch[1]
    NUM_FEATURES = gs[0][0].shape[-1]
    G = len(gs)
    N = sum(g[0].shape[0] for g in gs)
    M = sum(g[1].shape[0] for g in gs)
    adj = np.zeros([N, N])
    src = np.zeros([M])
    tgt = np.zeros([M])
    Msrc = np.zeros([N, M])
    Mtgt = np.zeros([N, M])
    Mgraph = np.zeros([N, G])
    h = np.concatenate([g[0] for g in gs])

    n_acc = 0
    m_acc = 0
    for g_idx, g in enumerate(gs):
        n = g[0].shape[0]
        m = g[1].shape[0]

        for e, (s, t) in enumerate(g[1]):
            adj[n_acc + s, n_acc + t] = 1
            adj[n_acc + t, n_acc + s] = 1

            src[m_acc + e] = n_acc + s
            tgt[m_acc + e] = n_acc + t

            Msrc[n_acc + s, m_acc + e] = 1
            Mtgt[n_acc + t, m_acc + e] = 1

        Mgraph[n_acc:n_acc + n, g_idx] = 1

        n_acc += n
        m_acc += m
    return (
        h.astype(NP_TORCH_FLOAT_DTYPE),
        adj.astype(NP_TORCH_FLOAT_DTYPE),
        src.astype(NP_TORCH_LONG_DTYPE),
        tgt.astype(NP_TORCH_LONG_DTYPE),
        Msrc.astype(NP_TORCH_FLOAT_DTYPE),
        Mtgt.astype(NP_TORCH_FLOAT_DTYPE),
        Mgraph.astype(NP_TORCH_FLOAT_DTYPE),
        labels
    )


def train(model, optimiser, dataset_loader, use_cuda, batch_size=1, disable_tqdm=False, profile=False):
    train_losses = []
    train_accs = []
    for b in tqdm(dataset_loader, desc="Instances ", disable=disable_tqdm):
        ta = time.time()
        optimiser.zero_grad()
        tb = time.time()
        h, adj, src, tgt, Msrc, Mtgt, Mgraph, np_labels = b
        tc = time.time()
        h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels = map(torch.from_numpy,
                                                               (h, adj, src, tgt, Msrc, Mtgt, Mgraph, np_labels))
        td = time.time()
        if use_cuda:
            h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels = map(to_cuda,
                                                                   (h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels))
        te = time.time()
        y = model(h, adj, src, tgt, Msrc, Mtgt, Mgraph)
        tf = time.time()
        loss = F.cross_entropy(input=y, target=pyt_labels)

        pred = torch.argmax(y, dim=1).detach().cpu().numpy()
        acc = np.sum((pred == np_labels).astype(float)) / pyt_labels.shape[0]
        mode = sp.stats.mode(pred)
        tg = time.time()

        tqdm.write(
            "{loss:.4f}\t{acc:.2f}%\t{mode} (x{modecount})".format(
                loss=loss.item(),
                acc=100 * acc,
                mode=mode[0][0],
                modecount=mode[1][0],
            )
        )

        th = time.time()
        loss.backward()
        optimiser.step()

        train_losses.append(loss.detach().cpu().item())
        train_accs.append(acc)
        if profile:
            ti = time.time()

            tt = ti - ta
            tqdm.write(
                "zg {zg:.2f}% bg {bg:.2f}% tt {tt:.2f}% tc {tc:.2f}% mo {mo:.2f}% me {me:.2f}% bk {bk:.2f}%".format(
                    zg=100 * (tb - ta) / tt,
                    bg=100 * (tc - tb) / tt,
                    tt=100 * (td - tc) / tt,
                    tc=100 * (te - td) / tt,
                    mo=100 * (tf - te) / tt,
                    me=100 * (tg - tf) / tt,
                    bk=100 * (ti - th) / tt,
                ))

    return train_losses, train_accs


def test(model, dataset_loader, use_cuda, desc="Test ", disable_tqdm=False):
    test_accs = []
    for b in tqdm(dataset_loader, desc=desc, disable=disable_tqdm):
        with torch.no_grad():
            h, adj, src, tgt, Msrc, Mtgt, Mgraph, np_labels = b
            h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels = map(torch.from_numpy,
                                                                   (h, adj, src, tgt, Msrc, Mtgt, Mgraph, np_labels))
            if use_cuda:
                h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels = map(to_cuda, (
                h, adj, src, tgt, Msrc, Mtgt, Mgraph, pyt_labels))

            y = model(h, adj, src, tgt, Msrc, Mtgt, Mgraph)

            pred = torch.argmax(y, dim=1).detach().cpu().numpy()
            acc = np.sum((pred == np_labels).astype(float)) / np_labels.shape[0]

            test_accs.append(acc)
    return test_accs


# Baseline utils

def get_image(PIL_image, desired_nodes=75):
    image = np.asarray(PIL_image).copy()
    image = image.astype(NP_TORCH_FLOAT_DTYPE) / 256.
    return image.transpose([2, 0, 1])


def get_supersegmented_image(PIL_image, desired_nodes=75):
    # load the image and convert it to a floating point data type
    image = np.asarray(PIL_image).copy()
    segments = slic(image, n_segments=desired_nodes, slic_zero=True)
    image = image.astype(NP_TORCH_FLOAT_DTYPE) / 256.
    asegments = np.array(segments)
    num_segments = np.max(asegments) + 1
    rgb_lists = [[] for i in range(num_segments)]

    height = image.shape[0]
    width = image.shape[1]
    for y in range(height):
        for x in range(width):
            node = asegments[y, x]
            rgb = image[y, x, :]
            rgb_lists[node].append(rgb)
        # end for
    # end for

    rgb_means = [np.mean(rgb_list, axis=0) for rgb_list in rgb_lists]

    for y in range(height):
        for x in range(width):
            node = asegments[y, x]
            rgb = rgb_means[node]
            image[y, x, :] = rgb[:]

    return image.transpose([2, 0, 1])


class ImgToGraphTransform:
    """Create two crops of the same image"""
    def __init__(self, desired_nodes=50):
        self.desired_nodes = desired_nodes

    def __call__(self, x):
        pil_image_transform = transforms.ToPILImage()
        return(get_graph_from_image(pil_image_transform(x), self.desired_nodes))


def graph_collate(batch):
    graph = [item[0] for item in batch]
    label = [item[1] for item in batch]
    label = torch.LongTensor(label)
    return [graph, label]

if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms
    import sys

    np.set_printoptions(threshold=sys.maxsize)

    train_dir = r"C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\dev_images\train"
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    pil_image_transform = transforms.ToPILImage()
    graphs = []
    labels = []
    for example in train_dataset:
        img = example[0]
        graph = get_graph_from_image(pil_image_transform(img))
        plot_graph_from_image(pil_image_transform(img),50,None)
        plot_graph_from_image(pil_image_transform(img),60,None)
        plot_graph_from_image(pil_image_transform(img),70,None)
        plot_graph_from_image(pil_image_transform(img),80,None)
        plot_graph_from_image(pil_image_transform(img),90,None)
        plot_graph_from_image(pil_image_transform(img),100,None)
        graphs.append(graph)
        labels.append(example[1])
        break
    print(train_dataset[0])

    train_dataset = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            ImgToGraphTransform(75)
        ])
    )
    print(train_dataset[0])
    # main(True,False,r"C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\dev_images\train\AK")
