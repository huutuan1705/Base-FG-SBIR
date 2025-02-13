import torch
import torch.nn as nn
import torch.nn.functional as F
import encoding # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionImage(nn.Module):
    def __init__(self, input_size=2048, output_size=64):
        super(AttentionImage, self).__init__()  
        self.head_layer = nn.Sequential(
            encoding.nn.Normalize(),
            nn.Linear(input_size, output_size),
            encoding.nn.Normalize()
        )

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        embedding = self.head_layer(x)
        
        return embedding
    
# input_tensor = torch.randn(68, 2048, 8, 8)
# model = AttentionImage(input_size=2048)
# output= model(input_tensor)

# print("Output shape:", output.shape)