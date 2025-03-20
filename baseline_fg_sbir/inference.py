import torch
import argparse
import numpy as np
import torch.utils.data as data 
import matplotlib.pyplot as plt

from dataset import FGSBIR_Dataset
from model import FGSBIR_Model
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    dataset_train = FGSBIR_Dataset(args, mode='train')
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))
    
    dataset_test = FGSBIR_Dataset(args, mode='test')
    dataloader_test = data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))
    
    return dataloader_train, dataloader_test

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

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Base Fine-Grained SBIR model')
    parsers.add_argument('--dataset_name', type=str, default='ShoeV2')
    parsers.add_argument('--backbone_name', type=str, default='InceptionV3', help='VGG16/InceptionV3/ResNet50')
    parsers.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parsers.add_argument('--output_size', type=int, default=64)
    parsers.add_argument('--num_heads', type=int, default=4)
    parsers.add_argument('--root_dir', type=str, default='./../')
    parsers.add_argument('--backbone_pretrained', type=str, default='./../')
    parsers.add_argument('--load_backbone_pretrained', type=bool, default=False)
    parsers.add_argument('--attention_pretrained', type=str, default='./../')
    parsers.add_argument('--linear_pretrained', type=str, default='./../')
    parsers.add_argument('--pretrained', type=str, default='./../')
    
    parsers.add_argument('--is_train', type=bool, default=True)
    parsers.add_argument('--load_pretrained', type=bool, default=True)
    parsers.add_argument('--train_backbone', type=bool, default=True)
    parsers.add_argument('--use_attention', type=bool, default=True)
    parsers.add_argument('--use_linear', type=bool, default=True)
    parsers.add_argument('--use_kaiming_init', type=bool, default=True)
    
    parsers.add_argument('--batch_size', type=int, default=16)
    parsers.add_argument('--test_batch_size', type=int, default=1)
    parsers.add_argument('--step_size', type=int, default=100)
    parsers.add_argument('--gamma', type=float, default=0.5)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--learning_rate', type=float, default=0.001)
    parsers.add_argument('--epochs', type=int, default=200)
    parsers.add_argument('--eval_freq_iter', type=int, default=100)
    parsers.add_argument('--print_freq_iter', type=int, default=1)
    
    args = parsers.parse_args()
    
    model = FGSBIR_Model(args=args)
    model.to(device)
    if args.load_pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    
    with torch.no_grad():
        model.eval()
        dataloader_train, dataloader_test = get_dataloader(args=args)
        count = 1
        for _, batch_data in enumerate(dataloader_train):
            visualize_layernorm(model, batch_data, count)
            break
            