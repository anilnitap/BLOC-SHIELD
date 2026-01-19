import torch
import torch.nn as nn

class CNNIDS(nn.Module):
    def __init__(self, input_dim):
        super(CNNIDS, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * ((input_dim - 2) // 2), 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
