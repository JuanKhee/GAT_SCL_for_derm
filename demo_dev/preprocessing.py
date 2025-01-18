import torchvision.transforms as transforms
import torch
import torchvision
import PIL
import matplotlib.pyplot as plt

train_root_path=r'dev_images\train'

train_transform = [
    transforms.RandomResizedCrop(size=(255, 255), scale=(0.2, 1.)),
      transforms.CenterCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.RandomApply([
          transforms.ColorJitter(0, 0, 0.2, 0)
      ], p=0.2),
      transforms.RandomPerspective(distortion_scale=0.1, p=0.4),
      transforms.RandomRotation(degrees=(0, 180)),
      transforms.ToTensor()
]

train_dataset = torchvision.datasets.ImageFolder(
    root=train_root_path,
    transform=transforms.ToTensor()
)

resize_transform = transforms.RandomResizedCrop(size=(255,255), scale=(0.6,1))
color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.02)
random_perspective = transforms.RandomPerspective(distortion_scale=0.2, p=0.4)
random_rotation = transforms.RandomRotation(degrees=(0, 180))
center_crop = transforms.CenterCrop(224)
all_transforms = [
    resize_transform,
    color_jitter,
    random_perspective,
    random_rotation,
    center_crop,
]
tensor_img = train_dataset[0][0]

print(tensor_img.shape)
plt.imshow(transforms.ToPILImage()(tensor_img))
plt.show()
plt.imshow(transforms.Resize((255,255))(tensor_img).permute(1,2,0))
plt.show()
plt.imshow(transforms.Compose([transforms.Resize((255,255)), transforms.CenterCrop((224,224))])(tensor_img).permute(1,2,0))
plt.show()

for i, transform in enumerate(all_transforms):
    fig, ax = plt.subplots(1,4)
    ax[0].imshow(transforms.ToPILImage()(tensor_img))
    ax[0].axis("off")
    for example in range(3):
        # transformed_image = transforms.Compose(all_transforms[:i + 1])(tensor_img)
        transformed_image = all_transforms[i+1](tensor_img)
        ax[example+1].imshow(transformed_image.permute(1,2,0))
        ax[example+1].axis("off")
    plt.show()
