import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from torchvision.models import Inception_V3_Weights, ResNet50_Weights, VGG16_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNet50(nn.Module):
    def __init__(self, args):
        super(ResNet50, self).__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)
                
        self.pool_method = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, input):
        x = self.features(input)
        x = self.pool_method(x)
        x = torch.flatten(x, 1)
        return F.normalize(x)
    
class VGG16(nn.Module):
    def __init__(self, args):
        super(VGG16, self).__init__()
        self.backbone = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        self.pool_method =  nn.AdaptiveMaxPool2d(1)

    def forward(self, input):
        x = self.backbone(input)
        x = self.pool_method(x).view(-1, 512)
        return F.normalize(x)
    
class InceptionV3(nn.Module):
    def __init__(self, args):
        super(InceptionV3, self).__init__()
        if args.load_pretrained==False:
            backbone = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        else:
            backbone = models.inception_v3()
        # backbone = models.inception_v3()
        
        ## Extract Inception Layers ##
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e

        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c
        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
        self.args = args

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        
        if self.args.use_attention:
            return x
        
        feature = self.pool_method(x).view(-1, 2048)
        return F.normalize(feature)
        
    def fix_weights(self):
        for x in self.parameters():
            x.requires_grad = False

class FeatureExtractor:
    def __init__(self, model, target_layers):
        """
        Lớp này giúp trích xuất feature maps từ các layer cụ thể trong mô hình
        
        Args:
            model: Mô hình cần trích xuất feature maps
            target_layers: Dictionary chứa tên và reference tới các layer cần theo dõi
        """
        self.model = model
        self.target_layers = target_layers
        self.features = {name: None for name in target_layers.keys()}
        self.hooks = []
        self._register_hooks()
        
    def _hook_fn(self, name):
        """Tạo một hook function để lưu output của layer"""
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook
        
    def _register_hooks(self):
        """Đăng ký các forward hooks cho các layer đã chọn"""
        for name, layer in self.target_layers.items():
            hook = layer.register_forward_hook(self._hook_fn(name))
            self.hooks.append(hook)
            
    def remove_hooks(self):
        """Xóa tất cả các hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def __call__(self, x):
        """Thực hiện forward pass qua mô hình và lấy các feature maps"""
        _ = self.model(x)
        return self.features
               
# dummy_input = torch.randn(25, 3, 299, 299)
# model = InceptionV3(None)
# output = model(dummy_input)
# print(output.shape) # [25, 2048]