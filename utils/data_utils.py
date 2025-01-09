import pandas as pd

# Makes sure we see all columns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import torch
from torch.utils.data import random_split


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


import torch
from tqdm import tqdm


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


def compute_mean_std(loader, size, dual=False):
    mean = 0.0
    print('calculating mean', flush=True)
    for inputs, _ in tqdm(loader):
        if dual:
            images = torch.tensor([inp[0].numpy() for inp in inputs])
        else:
            images = inputs
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    #var requires mean value, calculate after mean
    print('calculating std', flush=True)
    for inputs, _ in tqdm(loader):
        if dual:
            images = torch.tensor([inp[0].numpy() for inp in inputs])
        else:
            images = inputs
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(loader.dataset) * size * size))

    return(mean,std)