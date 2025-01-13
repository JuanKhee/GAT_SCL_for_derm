import os
import cv2
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torch

from utils.image import image_utils
from utils.data_utils import preprocess_images
from utils.graph_utils import batch_graphs, get_graph_from_image


def list_images(dev_image_path, metadata, id_col):
    dev_image = []
    for image in metadata[id_col]:
        if os.path.exists(os.path.join(dev_image_path, image + '.jpg')):
            dev_image.append(image)

    return dev_image


def split_folder_by_label(image_dir, metadata, id_col, label_col):
    output_image_dir = []
    for image_path in tqdm(os.listdir(image_dir)):
        label_df = metadata[metadata[id_col] == image_path.split('.')[0]][label_col]
        if(len(label_df) > 0):
            label = label_df.item()
            if not os.path.exists(os.path.join(image_dir, label)):
                os.makedirs(os.path.join(image_dir, label), exist_ok=True)
                output_image_dir.append(os.path.join(image_dir, label))
            shutil.move(os.path.join(image_dir,image_path), os.path.join(image_dir, label, image_path))

    return output_image_dir


def sparse_label_to_col(metadata, id_col, label_cols=('AK','BCC','BKL','DF','MEL','NV','SCC','VASC')):
    def extract_label(row):
        row_dict = row.to_dict()
        label = 'unknown'
        label_counter = 0
        for k,v in row_dict.items():
            if k in label_cols:
                if v == 1: label = k
                label_counter += int(v)

        if label_counter > 1: print(row[id_col], 'has more than 1 diagnosis')

        return label

    metadata['label'] = metadata.apply(extract_label, axis=1)

    return metadata


def prepare_dataset_dir(img_path, metadata_path, id_col):
    metadata = pd.read_csv(metadata_path)

    metadata_processed = sparse_label_to_col(metadata, id_col)
    counter_df = metadata_processed.groupby('label').count()[id_col].reset_index()
    counter_df['percentage'] = counter_df[id_col] / counter_df[id_col].sum()
    print(counter_df)
    metadata_processed.to_csv(f'{metadata_path.split(".")[0]}_processed.csv')

    images = list_images(img_path, metadata_processed, id_col)
    data_dir = split_folder_by_label(img_path, metadata_processed, id_col, 'label')

    return images, data_dir


def prepare_graphs(dataset, nodes=50, output_dir="Training_Graphs_50_nodes"):
    for i in tqdm(range(len(dataset))):
        img = dataset[i][0]
        pil_image_transform = transforms.ToPILImage()
        graph = get_graph_from_image(pil_image_transform(img), nodes)
        input_path = dataset.imgs[i][0]
        input_dirs = input_path.split(os.sep)
        output_path = os.sep.join(input_dirs[:-3])
        output_dir = output_dir
        full_output_dir = os.sep.join([output_path, output_dir, input_dirs[-2]])
        output_path = os.path.join(full_output_dir, input_dirs[-1].split('.')[0])
        os.makedirs(full_output_dir, exist_ok=True)

        np.save(output_path, graph, allow_pickle=True)
    return True


if __name__ == "__main__":
    # dev run
    # train_img_path = 'dev_images/train'
    # train_metadata_path = 'metadata/ISIC_2019_Training_GroundTruth.csv'
    # test_img_path = 'dev_images/test'
    # test_metadata_path = 'metadata/ISIC_2019_Test_GroundTruth.csv'
    # id_col = 'image'

    # actual data run
    train_img_path = r"C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\dev_images\train"
    train_metadata_path = 'metadata/ISIC_2019_Training_GroundTruth.csv'
    test_img_path = r"C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\dev_images\test"
    test_metadata_path = 'metadata/ISIC_2019_Test_GroundTruth.csv'
    id_col = 'image'

    # train_img_path = r"C:\Users\HP-VICTUS\Documents\Masters\WQF7023 AI Project\dataset\ISIC_2019_Training_Input\ISIC_2019_Training_Input"
    # train_metadata_path = 'metadata/ISIC_2019_Training_GroundTruth.csv'
    # test_img_path = r"C:\Users\HP-VICTUS\Documents\Masters\WQF7023 AI Project\dataset\ISIC_2019_Test_Input\ISIC_2019_Test_Input"
    # test_metadata_path = 'metadata/ISIC_2019_Test_GroundTruth.csv'
    # id_col = 'image'

    # train_images, train_data_dir = prepare_dataset_dir(train_img_path, train_metadata_path, id_col)
    # test_images, test_data_dir = prepare_dataset_dir(test_img_path, test_metadata_path, id_col)
    #
    # print(train_data_dir)
    # print(test_data_dir)
    # dev_img = []
    # for img_file in dev_images:
    #     img_path =  os.path.join(dev_train_img_path,img_file+'.jpg')
    #     img = cv2.imread(img_path)
    #     dev_img.append(img)
    #     image_utils.print_image_detail(img)

    train_dataset = torchvision.datasets.ImageFolder(
        root=train_img_path,
        transform=transforms.ToTensor(),
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=test_img_path,
        transform=transforms.ToTensor(),
    )

    print(prepare_graphs(train_dataset, nodes=60, output_dir="Training_Graph_60_nodes"))
    print(prepare_graphs(test_dataset, nodes=60, output_dir="Test_Graph_60_nodes"))
