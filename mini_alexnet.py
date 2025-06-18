import torch.nn as nn

class MiniAlexNet(nn.Module):
    def __init__(self):
        super(MiniAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 12 * 12, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # paper, rock, scissors
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x