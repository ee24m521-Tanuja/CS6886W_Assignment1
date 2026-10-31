import torch.nn as nn
class SiLUActivation(nn.Module):
    def __init__(self):
        super().__init__(); self.op = nn.SiLU(inplace=True)
    def forward(self, x): return self.op(x)
