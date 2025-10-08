import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(24, 12)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(12, 6)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(6, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        return x


class ClassificationModel(nn.Module):
    def __init__(self, input_dim: int, outout_dim: int, dropout_prob: int=0.3): 
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.fc2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(128, outout_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        s = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x