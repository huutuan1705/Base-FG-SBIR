import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

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
    # Danh sách các layer cần visualize
    layers = ['Conv2d_1a_3x3', 'Conv2d_2b_3x3', 'Conv2d_4a_3x3', 'Mixed_6a', 'Mixed_6e', 'Mixed_7c']
    
    plt.figure(figsize=(20, 30))
    rows = len(layers) * 2  # 2 loại: sketch và positive
    
    for i, layer_name in enumerate(layers):
        # Lấy feature maps
        print(sketch_features)
        sketch_map = sketch_features[layer_name][0].detach().cpu()  # Lấy batch đầu tiên
        positive_map = positive_features[layer_name][0].detach().cpu()
        
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

def visualize_batch(model, batch, save_dir='visualizations'):
    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(save_dir, exist_ok=True)
    
    # Chuyển mô hình sang chế độ đánh giá
    model.eval()
    
    # Forward pass với việc trả về các feature maps
    with torch.no_grad():
        sketch_features, positive_features = model(batch, return_features=True)
    
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