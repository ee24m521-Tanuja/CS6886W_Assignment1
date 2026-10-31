import torch.nn as nn
class SigmoidActivation(nn.Module):
    def __init__(self):
        super().__init__(); self.op = nn.Sigmoid()
    def forward(self, x): return self.op(x)
