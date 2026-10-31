import torch.nn as nn
class GELUActivation(nn.Module):
    def __init__(self):
        super().__init__(); self.op = nn.GELU()
    def forward(self, x): return self.op(x)
