import torch.nn as nn
import torch
import os
import sys
sys.path.append(os.getcwd())

#####################################################################
# NUS-WIDE
class MLP2(nn.Module):
    def __init__(self, input_dim, dropout_rate):
        super(MLP2, self).__init__()
        hidden_dims=[512, 512]

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)         
        return x
