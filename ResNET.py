import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Linear(in_channels, out_channels, bias=False)
        self.bn1 = nn.LayerNorm(out_channels)  # layer normalization on last dimension
        
        self.conv2 = nn.Linear(out_channels, out_channels, bias=False)
        self.bn2 = nn.LayerNorm(out_channels)  # layer normalization on last dimension
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.LayerNorm(out_channels)  # matching normalization for shortcut
            )
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)  # normalize over the last dimension

        out = self.conv2(out)
        out = self.bn2(out)  # normalize over the last dimension
        
        out += self.shortcut(x)  # add the shortcut connection
        
        return out
