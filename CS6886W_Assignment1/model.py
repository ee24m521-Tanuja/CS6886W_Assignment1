import math, torch
import torch.nn as nn
import torch.nn.functional as F
from activations import get_activation
CFG=[64,64,'M',128,128,'M']
class VGG6(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__(); layers=[]; c=3
        for v in CFG:
            if v=='M': layers.append(nn.MaxPool2d(2,2))
            else: layers += [nn.Conv2d(c,v,3,padding=1), nn.BatchNorm2d(v), get_activation(activation)]; c=v
        self.features=nn.Sequential(*layers); self.classifier=nn.Linear(c,10); self._initialize()
    def forward(self,x):
        x=self.features(x); x=F.adaptive_avg_pool2d(x,(1,1)); return self.classifier(torch.flatten(x,1))
    def _initialize(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels; m.weight.data.normal_(0,math.sqrt(2./n));
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d): m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m,nn.Linear): nn.init.kaiming_normal_(m.weight,nonlinearity='relu'); m.bias.data.zero_()
