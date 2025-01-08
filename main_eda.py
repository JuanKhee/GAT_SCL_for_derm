import torch
import torchvision
import torchvision.transforms as transforms
from utils.data_utils import compute_mean_std

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