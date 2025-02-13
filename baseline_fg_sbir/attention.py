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
    
# input_tensor = torch.randn(68, 2048, 8, 8)
# model = AttentionImage(input_size=2048)
# output= model(input_tensor)

# print("Output shape:", output.shape)