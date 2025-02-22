import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Attention_global(nn.Module):
    def __init__(self):
        super(Attention_global, self).__init__()
        self.pool_method =  nn.AdaptiveAvgPool2d(1) # as default
        self.net = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(),
                                 nn.Conv2d(512, 1, kernel_size=1))
       
    def forward(self, backbone_tensor):
        backbone_tensor_1 = self.net(backbone_tensor)
        backbone_tensor_1 = backbone_tensor_1.view(backbone_tensor_1.size(0), -1)
        backbone_tensor_1 = nn.Softmax(dim=1)(backbone_tensor_1)
        backbone_tensor_1 = backbone_tensor_1.view(backbone_tensor_1.size(0), 1, backbone_tensor.size(2), backbone_tensor.size(3))
        fatt = backbone_tensor*backbone_tensor_1
        fatt1 = backbone_tensor +fatt
        fatt1 = self.pool_method(fatt1).view(-1, 2048)
        return  F.normalize(fatt1)

class Linear_global(nn.Module):
    def __init__(self, feature_num):
        super(Linear_global, self).__init__()
        self.head_layer = nn.Linear(2048, feature_num)
    
    def forward(self, x):
        return F.normalize(self.head_layer(x))
        # return fatt1

class AttentionBlock(nn.Module):
    def __init__(self, in_channels=2048):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        identity = x  # Lưu lại đầu vào
        out = self.conv1(x)
        out = self.bn(out)
        out = self.conv2(out)
        out = self.softmax(out)

        out = out * identity  # dot product
        x = out + identity # residual

        # Global pooling
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x 
    
# input_tensor = torch.randn(68, 2048, 8, 8)
# model = Attention_global()
# output= model(input_tensor)

# print("Output shape:", output.shape)