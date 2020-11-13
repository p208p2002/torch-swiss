import torch
import torch.nn as nn
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(32,2)
    
    def forward(x):
        return self.layer(x)