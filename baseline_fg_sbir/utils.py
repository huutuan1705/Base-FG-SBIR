import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from collections import OrderedDict
from backbones import FeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transform(type):
    if type == 'train':
        transform_list = [
            transforms.Resize(299),
            transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        
    else: 
        transform_list = [
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        
    return transforms.Compose(transform_list)

def visualize_feature_maps(sketch_features, positive_features, save_path='feature_maps.png'):
    """
    Visualize các feature maps từ các layer được chỉ định và lưu thành một ảnh duy nhất
    """
    # Danh sách các layer cần visualize
    layer_names = list(sketch_features.keys())
    
    plt.figure(figsize=(20, 30))
    rows = len(layer_names) * 2  # 2 loại: sketch và positive
    
    for i, layer_name in enumerate(layer_names):
        # Lấy feature maps
        sketch_map = sketch_features[layer_name][0].cpu()  # Lấy batch đầu tiên
        positive_map = positive_features[layer_name][0].cpu()
        
        # Chọn một số kênh để hiển thị (ví dụ: 8 kênh đầu tiên hoặc tất cả nếu ít hơn 8)
        num_channels = min(8, sketch_map.shape[0])
        
        # Hiển thị feature maps của sketch
        for j in range(num_channels):
            plt.subplot(rows, num_channels, i*2*num_channels + j + 1)
            plt.imshow(sketch_map[j].numpy(), cmap='viridis')
            if j == 0:
                plt.ylabel(f'Sketch\n{layer_name}', fontsize=12)
            plt.xticks([])
            plt.yticks([])
        
        # Hiển thị feature maps của positive image
        for j in range(num_channels):
            plt.subplot(rows, num_channels, (i*2+1)*num_channels + j + 1)
            plt.imshow(positive_map[j].numpy(), cmap='viridis')
            if j == 0:
                plt.ylabel(f'Positive\n{layer_name}', fontsize=12)
            plt.xticks([])
            plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature maps saved to {save_path}")

def visualize_with_hooks(model, batch, save_dir='visualizations'):
    """
    Sử dụng hooks để visualize feature maps từ một batch
    """
    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(save_dir, exist_ok=True)
    
    # Chuyển mô hình sang chế độ đánh giá
    model.eval()
    
    # Xác định các layer cần visualize
    sketch_network = model.sketch_embedding_network
    photo_network = model.sample_embedding_network
    
    sketch_target_layers = OrderedDict([
        ('Conv2d_1a_3x3', sketch_network.Conv2d_1a_3x3),
        ('Conv2d_2b_3x3', sketch_network.Conv2d_2b_3x3),
        ('Conv2d_4a_3x3', sketch_network.Conv2d_4a_3x3),
        ('Mixed_5d', sketch_network.Mixed_5d),
        ('Mixed_6d', sketch_network.Mixed_6d),
        ('Mixed_7c', sketch_network.Mixed_7c)
    ])
    
    photo_target_layers = OrderedDict([
        ('Conv2d_1a_3x3', photo_network.Conv2d_1a_3x3),
        ('Conv2d_2b_3x3', photo_network.Conv2d_2b_3x3),
        ('Conv2d_4a_3x3', photo_network.Conv2d_4a_3x3),
        ('Mixed_5d', photo_network.Mixed_5d),
        ('Mixed_6d', photo_network.Mixed_6d),
        ('Mixed_7c', photo_network.Mixed_7c)
    ])
    
    # Tạo feature extractors
    sketch_extractor = FeatureExtractor(sketch_network, sketch_target_layers)
    photo_extractor = FeatureExtractor(photo_network, photo_target_layers)
    
    # Trích xuất feature maps
    with torch.no_grad():
        sketch_img = batch['sketch_img'].to(device)
        positive_img = batch['positive_img'].to(device)
        
        sketch_features = sketch_extractor(sketch_img)
        positive_features = photo_extractor(positive_img)
    
    # Xóa hooks sau khi sử dụng
    sketch_extractor.remove_hooks()
    photo_extractor.remove_hooks()
    
    # Visualize và lưu feature maps
    save_path = os.path.join(save_dir, f"feature_maps_batch.png")
    visualize_feature_maps(sketch_features, positive_features, save_path)
    
    return save_path


def visualize_layernorm(model, sample_input, num=1):
    """ Hàm để visualize attention.norm và sketch_attention.norm """

    # Tạo biến lưu giá trị đầu vào & đầu ra của hai LayerNorm
    ln_inputs = {'attention.norm': [], 'sketch_attention.norm': []}
    ln_outputs = {'attention.norm': [], 'sketch_attention.norm': []}

    # Hàm hook để lưu dữ liệu
    def hook_fn(layer_name):
        def hook(module, input, output):
            ln_inputs[layer_name].append(input[0].detach().cpu().numpy().flatten())
            ln_outputs[layer_name].append(output.detach().cpu().numpy().flatten())
        return hook

    # Đặt hook vào hai LayerNorm
    hooks = []
    hooks.append(model.attention.norm.register_forward_hook(hook_fn('attention.norm')))
    hooks.append(model.sketch_attention.norm.register_forward_hook(hook_fn('sketch_attention.norm')))

    # Chạy inference
    with torch.no_grad():
        model(sample_input)

    # Gỡ hook sau khi lấy dữ liệu
    for hook in hooks:
        hook.remove()

    # Chuyển dữ liệu thành numpy
    for key in ln_inputs:
        ln_inputs[key] = np.concatenate(ln_inputs[key])
        ln_outputs[key] = np.concatenate(ln_outputs[key])

    # Vẽ hai biểu đồ scatter
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    for idx, key in enumerate(ln_inputs):
        axs[idx].scatter(ln_inputs[key], ln_outputs[key], s=1, alpha=0.5)
        axs[idx].set_xlabel("LN Input")
        axs[idx].set_ylabel("LN Output")
        axs[idx].set_title(f"Scatter plot for {key}")
        axs[idx].grid(True)
        
        axs[idx].set_xlim(-3, 14)
        axs[idx].set_ylim(-3, 14)

    name = "visualization" + str(num) + ".png"
    plt.tight_layout()
    plt.savefig(name)
    plt.show()