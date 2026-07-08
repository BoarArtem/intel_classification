from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

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
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# dataset exc.
train_dataset = ImageFolder(root="../data/seg_train/seg_train", transform=train_transform)
test_dataset = ImageFolder(root="../data/seg_test/seg_test", transform=test_transform)

# dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, num_workers=4, pin_memory=True)
