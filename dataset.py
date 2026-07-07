from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# data augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=35),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3
    ),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# dataset exc.
train_dataset = ImageFolder(root="../data/seg_train/seg_train", transform=train_transform)
test_dataset = ImageFolder(root="../data/seg_test/seg_test", transform=test_transform)

# dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64)
