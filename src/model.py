import torch
import torch.nn as nn

num_classes = 2
image_size = 224

class BinaryImageClassification(nn.Module):
    def __init__(self):
        super(BinaryImageClassification, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        self.bn3 = nn.BatchNorm2d(128)

        #print("Input size for fc1:", 128 * ((image_size // 8) - 1) * ((image_size // 8) - 1))

        self.fc1 = nn.Linear(128 * ((image_size // 8) - 1) * ((image_size // 8) - 1), 512)
        self.bnfc1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(nn.functional.max_pool2d(self.conv1(x), 2)))
        #print("Size after first conv", x.size())
        x = nn.functional.relu(self.bn2(nn.functional.max_pool2d(self.conv2(x), 2)))
        #print("Size after second conv", x.size())
        x = nn.functional.relu(self.bn3(nn.functional.max_pool2d(self.conv3(x), 2)))

        #print("Size before flattening", x.size())
        x = x.view(-1, self.num_flat_features(x))
        x = nn.functional.relu(self.bnfc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        return x.size(1) * x.size(2) * x.size(3)
