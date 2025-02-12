import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionImage(nn.Module):
    def __init__(self, input_size, hidden_layer=2048):
        super(AttentionImage, self).__init__()  
        self.input_size = input_size      
        self.attn_hidden_layer = hidden_layer

        self.net = nn.Sequential(
            nn.Conv2d(self.input_size, self.attn_hidden_layer, kernel_size=1), 
            nn.BatchNorm2d(self.attn_hidden_layer), 
            nn.ReLU(),
            nn.Conv2d(self.attn_hidden_layer, 1, kernel_size=1)
        )

        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x):
        attn_mask = self.net(x).to(device)  # (batch_size, 1, H, W)
        attn_mask = attn_mask.view(attn_mask.size(0), -1)  # reshape (batch_size, H*W)
        attn_mask = self.softmax(attn_mask)  # Softmax
        attn_mask = attn_mask.view(x.size(0), 1, x.size(2), x.size(3))  # reshape (batch_size, 1, H, W)

        x_attn = x * attn_mask  # attention
        x = x + x_attn  # Residual connection

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        return x
    
# input_tensor = torch.randn(68, 2048, 8, 8)
# model = AttentionImage(input_size=2048)
# output= model(input_tensor)

# print("Output shape:", output.shape)