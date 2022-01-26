import iresnet
from torch import nn
import numpy as np

class FocusFace(nn.Module):
    def __init__(self,identities=1000):
        super(FocusFace,self).__init__()
        self.model = iresnet.iresnet100()
        self.model.fc = EmbeddingHead(512,32)
        self.fc = ArcMarginProduct(512, identities, s=64,m=0.5) #m=0.35)
        self.fc2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x,inference=False):
        e1,e2 = self.model(x)
        y = None
        if not(inference):
            y = self.fc(e1.view(e1.shape[0],-1),label)
            y2 = self.fc2(e2.view(e2.shape[0],-1))
            e2 = e2.view(e2.shape[0],-1)
        e1 = e1.view(e1.shape[0],-1)
        if inference: 
            y2 = self.fc2(e2.view(e2.shape[0],-1))
        if not(inference):
            return y,e1,e2,y2
        return None,e1,None,torch.nn.functional.softmax(y2)[:,1]



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

    
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp((1.0 - torch.pow(cosine, 2)),1e-9,1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=label.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
