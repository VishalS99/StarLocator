import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class StarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 32, 3, padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.classification = nn.Sequential(
            nn.Linear(32*12*12, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.regression = nn.Sequential(
            nn.Linear(32*12*12, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 5),
        )



    def forward(self, x, classification=False):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        x = x + self.conv6(x)
        x = self.conv7(x)
        # size: (64,32,12,12)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        if classification:
            class_x = self.classification(x)
            return class_x
        reg_x = self.regression(x)
        return torch.sigmoid(reg_x)

    def pred(self, x):
        prediction = self.forward(x, True)
        if prediction > 0.5:
            output = torch.squeeze(self.forward(x, False)).detach().cpu()
            return output
        return torch.from_numpy(np.full(5, np.nan))
