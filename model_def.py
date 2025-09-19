import torch
import torch.nn as nn

class ACEPnet(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
        
        self.feature_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        
        self.final = nn.Sequential(
            nn.Linear(128 * 8 * 8 + 128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
    def forward(self, img, features):
        x = self.cnn(img)
        att = self.attention(x)
        x = x * att
        x = x.view(x.size(0), -1)
        
        f = self.feature_net(features)
        combined = torch.cat([x, f], dim=1)
        out = self.final(combined)
        return out
