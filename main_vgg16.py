import numpy as np
import sys
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from SkinDiseaseClassifierCNN import SkinDiseaseClassifier
from datetime import datetime
from models.CNN import CNNModel

np.set_printoptions(threshold=sys.maxsize)

vgg16_model = CNNModel(8)
dev_classifier = SkinDiseaseClassifier(
    vgg16_model,
    epochs=20,
    batch_size=32,
    learning_rate=0.0001,
    output_dir='model_result_vgg16'
)

train_transform = [
        # transforms.RandomResizedCrop(size=(255, 255), scale=(0.2, 1.)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        # ], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        transforms.Resize((255,255)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.6678, 0.5298, 0.5245], std=[0.2232, 0.2030, 0.2146])
    ]

test_transform = [
                transforms.Resize((255, 255)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.6678, 0.5298, 0.5245], std=[0.2232, 0.2030, 0.2146])
            ]

dev_classifier.create_dataloader(
    train_root_path=r"C:\Users\HP-VICTUS\Documents\Masters\WQF7023 AI Project\dataset\ISIC_2019_Training_Input\ISIC_2019_Training_Input",
    test_root_path=r"C:\Users\HP-VICTUS\Documents\Masters\WQF7023 AI Project\dataset\ISIC_2019_Test_Input\ISIC_2019_Test_Input",
    seed=57,
    train_transform=train_transform,
    test_transform=test_transform
)
dev_classifier.cross_validate(5,57,'test_for_mean',None)
# train_start_time = datetime.now()
# dev_classifier.train_model()
# train_end_time = datetime.now()
# dev_classifier.load_model()
# dev_classifier.evaluate_model()

# print(f'''
# start_time:{train_start_time}
# end_time  :{train_end_time}
# time_taken:{(train_end_time - train_start_time).seconds}
# ''')