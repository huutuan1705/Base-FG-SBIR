import torch
import argparse
import numpy as np
import torch.utils.data as data 

from dataset import FGSBIR_Dataset
from model import FGSBIR_Model
from utils import visualize_layernorm, visualize_with_hooks
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    dataset_train = FGSBIR_Dataset(args, mode='train')
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))
    
    dataset_test = FGSBIR_Dataset(args, mode='test')
    dataloader_test = data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))
    
    return dataloader_train, dataloader_test

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
    
    dataloader_train, dataloader_test = get_dataloader(args=args)
    
    for _, batch_data in enumerate(dataloader_train):
        visualize_with_hooks(model, batch_data)
        break
            