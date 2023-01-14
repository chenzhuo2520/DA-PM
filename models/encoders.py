import torch
from torch import nn

class My_ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(768, 1)

    def forward(self, text_features):
        # x = text_features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(text_features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Full_classificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768*3, 768*3)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(768*3, 1)

    def forward(self, text_features):
        # x = text_features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(text_features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x