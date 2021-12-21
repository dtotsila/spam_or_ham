import torch
import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """ Custom Spam or Ham Dataset. """

    def __init__(self, x, y):
        """
        Args:
            x (numpy array): # tf-idf vector representation of message
            y (numpy array): # class spam or ham
        """
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



# Simple Logistic Regresion Module (sigmoid activated)
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)     
    
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs
