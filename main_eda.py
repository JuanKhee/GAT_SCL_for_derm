import torch
import torchvision
import torchvision.transforms as transforms
from utils.data_utils import compute_mean_std
import numpy as np

dataset = torchvision.datasets.ImageFolder(
    r"C:\Users\HP-VICTUS\Documents\Masters\WQF7023 AI Project\dataset\ISIC_2019_Training_Input\ISIC_2019_Training_Input",
    transform=transforms.Compose([transforms.Resize((255,255)), transforms.ToTensor()])
)

loader = torch.utils.data.DataLoader(dataset,
                         batch_size=10,
                         num_workers=0,
                         shuffle=False)

mean,std = compute_mean_std(loader, 255)
print(mean,std)
#[0.6678, 0.5298, 0.5245], [0.2232, 0.2030, 0.2146]

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

counts = np.array(list(cat_counts.values()))
class_weights = max(counts)/counts
print(class_weights)
# [0.06733981 0.25809709 0.20380583 0.01856311 0.3512233  1.
#  0.0487767  0.01965049]
