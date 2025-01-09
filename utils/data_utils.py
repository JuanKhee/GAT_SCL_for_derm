import numpy as np
import pandas as pd
from torch.utils.data import random_split
import multiprocessing
import torch
from tqdm import tqdm

# Makes sure we see all columns
pd.set_option('display.max_columns', None)


def split(full_dataset, val_percent, random_seed=None):
    amount = len(full_dataset)

    val_amount = (
        int(amount * val_percent)
        if val_percent is not None else 0)
    train_amount = amount - val_amount

    train_dataset, val_dataset = random_split(
        full_dataset,
        (train_amount, val_amount),
        generator=(
            torch.Generator().manual_seed(random_seed)
            if random_seed
            else None))

    return train_dataset, val_dataset


def normalise_pixel_value(img):
    img = img/255

    return img


def preprocess_images(images):
    images_processed = []
    for img in images:
        print(img)
        img_processed = normalise_pixel_value(img)
        images_processed.append(img_processed)

    return images_processed

def compute_sums(images, batch_samples):
    return images.view(batch_samples, images.size(1), -1).mean(2).sum(0)


def compute_sum_vars(images, batch_samples, mean):
    return ((images.view(batch_samples, images.size(1), -1) - mean.unsqueeze(1)) ** 2).sum([0, 2])


def compute_mean_std(loader, size, dual=False, pools=1):
    sums = []
    print('calculating mean', flush=True)
    for inputs, _ in tqdm(loader):
        if dual:
            images = torch.tensor([inp[0].numpy() for inp in inputs])
        else:
            images = inputs
        batch_samples = images.size(0)
        with multiprocessing.Pool(pools) as p:
            sums = np.array(p.map(compute_sums, images, batch_samples))
    mean = np.sum(sums, axis=0)/len(loader.dataset)

    sum_vars = []
    #var requires mean value, calculate after mean
    print('calculating std', flush=True)
    for inputs, _ in tqdm(loader):
        if dual:
            images = torch.tensor([inp[0].numpy() for inp in inputs])
        else:
            images = inputs
        batch_samples = images.size(0)

        with multiprocessing.Pool(pools) as p:
            sum_vars = np.array(
                p.map(
                    compute_sum_vars,
                    images,
                    batch_samples,
                    mean
                )
            )
    std = torch.sqrt(np.sum(sum_vars, axis=0) / (len(loader.dataset) * size * size))

    print(mean, std)
    return(mean,std)