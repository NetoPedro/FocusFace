import iresnet
from torch import nn
import numpy as np

class FocusFace(nn.Module):
    def __init__(self,identities=1000):
        super(FocusFace,self).__init__()
        self.model = iresnet.iresnet100()
        self.model.fc = EmbeddingHead(512,32)

    def forward(self, x):
        e1,_ = self.model(x)
        e1 = e1.view(e1.shape[0],-1)
        return e1



class EmbeddingHead(nn.Module):
    def __init__(self, c1=512,c2=256):
        super(EmbeddingHead,self).__init__()
        self.conv1 = nn.Conv2d(512, c1, kernel_size=(7, 7), stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(512, c2, kernel_size=(7, 7), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(c1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(c2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self,x):
        size = int(np.sqrt(x.shape[1]/512))
        x = x.view((x.shape[0],-1,size,size))
        return self.bn1(self.conv1(x)), self.relu(self.bn2(self.conv2(x)))
