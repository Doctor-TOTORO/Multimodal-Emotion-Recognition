import torch.nn as nn
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class SENet_Emotion_Net(nn.Module):

    def __init__(self, num_class):
        super(SENet_Emotion_Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            SELayer(256),
            nn.MaxPool1d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            SELayer(512),
            nn.MaxPool1d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            SELayer(512),
            nn.MaxPool1d(2, 2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(16384, 512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = x.reshape(len(x), 1, -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
