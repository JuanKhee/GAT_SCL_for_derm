import torch
import torchvision
import torchvision.transforms as transforms
from utils.data_utils import compute_mean_std
import numpy as np
import pandas as pd
import os

cat_counts = {
    "AK": 867,
    "BCC": 3323,
    "BKL": 2624,
    "DF": 239,
    "MEL": 4522,
    "NV": 12875,
    "SCC": 628,
    "VASC": 253
}


def get_train_mean_std(loader):

    mean,std = compute_mean_std(loader, 255)
    return mean, std


def get_class_weights(cat_counts):
    counts = np.array(list(cat_counts.values()))
    class_weights = max(counts) / counts

    return class_weights


def get_unique_metadata_values(metadata, id_cols):
    col_unique_val = {}
    for col in metadata.columns:
        if col not in id_cols:
            col_unique_val[col] = list(metadata[col].unique())

    return col_unique_val


if __name__ == "__main__":
    from utils.data_utils import get_metadata_vector
    # dataset = torchvision.datasets.ImageFolder(
    #     r"C:\Users\HP-VICTUS\Documents\Masters\WQF7023 AI Project\dataset\ISIC_2019_Training_Input\ISIC_2019_Training_Input",
    #     transform=transforms.Compose([transforms.Resize((255, 255)), transforms.ToTensor()])
    # )
    #
    # loader = torch.utils.data.DataLoader(dataset,
    #                                      batch_size=10,
    #                                      num_workers=0,
    #                                      shuffle=False)
    #
    # mean,std = get_train_mean_std(loader)
    # print(mean,std)
    # # [0.6678, 0.5298, 0.5245], [0.2232, 0.2030, 0.2146]
    #
    # class_weights = get_class_weights(cat_counts)
    # print(class_weights)
    # # [0.06733981 0.25809709 0.20380583 0.01856311 0.3512233  1.
    # #  0.0487767  0.01965049]

    train_metadata = pd.read_csv(r"metadata\ISIC_2019_Training_Metadata.csv")
    train_metadata_unique_val = get_unique_metadata_values(train_metadata, ['image','lesion_id'])
    print(train_metadata_unique_val)
    # 'anatom_site_general': ['anterior torso', 'upper extremity', 'posterior torso', 'lower extremity', nan, 'lateral torso', 'head/neck', 'palms/soles', 'oral/genital'],
    # 'sex': ['female', 'male', nan]

    age_mean = np.mean(train_metadata['age_approx'])
    age_std = np.std(train_metadata['age_approx'])
    print(age_mean,age_std)

    site = sorted([s for s in train_metadata_unique_val['anatom_site_general'] if type(s) == str])
    print(site)
    # ['anterior torso', 'head/neck', 'lateral torso', 'lower extremity', 'oral/genital', 'palms/soles', 'posterior torso', 'upper extremity']

    sex = sorted([s for s in train_metadata_unique_val['sex'] if type(s) == str])
    print(sex)
    # ['female', 'male']

    print(get_metadata_vector(metadata=train_metadata, image_id='ISIC_0000000'))
    # [0.05358448431063522, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    dataset = torchvision.datasets.ImageFolder(
        r"C:\Users\HP-VICTUS\Documents\Masters\WQF7023 AI Project\dataset\ISIC_2019_Training_Input\ISIC_2019_Training_Input",
        transform=transforms.Compose([transforms.Resize((255, 255)), transforms.ToTensor()])
    )

    print(dataset.imgs[0][0].split(os.sep)[-1].split('.')[0])
    # ISIC_0024468
    print(get_metadata_vector(metadata=train_metadata, image_id=dataset.imgs[0][0].split(os.sep)[-1].split('.')[0]))
    # [1.1566914947956077, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]

