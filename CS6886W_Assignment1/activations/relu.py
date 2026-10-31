import torch.nn as nn
class ReLUActivation(nn.Module):
    def __init__(self):
        super().__init__(); self.op = nn.ReLU(inplace=True)
    def forward(self, x): return self.op(x)
