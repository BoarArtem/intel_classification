import torch
from torch import nn
from train import train
from test import test
from utils import get_optim, criterion
from dataset import train_loader, test_loader

class IntelModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),

            nn.Conv2d(in_channels=32, out_channels=32, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=64, out_channels=128, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.MaxPool2d(2),
        )

        self.fc_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 6)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = IntelModel().to(device)
optim = get_optim(model)

# train and test loop
if __name__ == "__main__":
    print("Training start successfully...")
    train(50, model, train_loader, device, optim, criterion)
    test(model, test_loader, device)