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


def compute_mean_std(loader, size):
    mean = 0.0
    print('calculating mean')
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    print('calculating std')
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(loader.dataset) * size * size))

    return(mean,std)