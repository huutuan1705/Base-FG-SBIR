import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionImage(nn.Module):
    def __init__(self, input_size=2048, output_size=64, eps=1e-12):
        super(AttentionImage, self).__init__()  
        self.eps = eps
        self.head_layer = nn.Linear(input_size, output_size)

    def normalize(self, x):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        return x / (norm + self.eps)
    
    def forward(self, x):
        x = self.head_layer(x)
        embedding = self.normalize(x)
        
        return embedding
    
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
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
# model = AttentionBlock(in_channels=2048)
# output= model(input_tensor)

# print("Output shape:", output.shape)