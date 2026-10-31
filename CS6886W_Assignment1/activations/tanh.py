import torch.nn as nn
class TanhActivation(nn.Module):
    def __init__(self):
        super().__init__(); self.op = nn.Tanh()
    def forward(self, x): return self.op(x)
