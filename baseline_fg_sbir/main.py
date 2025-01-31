import time
import torch
import argparse
import torch.utils.data as data 

from dataset import FGSBIR_Dataset

def get_dataloader(args):
    dataset_train = FGSBIR_Dataset(args, mode='train')
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))
    
    dataset_test = FGSBIR_Dataset(args, mode='test')
    dataloader_test = data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=int(args.threads))
    
    return dataloader_train, dataloader_test

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Base Fine-Grained SBIR model')
    parsers.add_argument('--dataset_name', type=str, default='ShoeV2')
    parsers.add_argument('--backbone_name', type=str, default='InceptionV3', help='VGG16/InceptionV3/Resnet50')
    parsers.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parsers.add_argument('--root_dir', type=str, default='./../')
    parsers.add_argument('--batch_size', type=int, default=16)
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--learning_rate', type=float, default=0.0001)
    parsers.add_argument('--max_epoch', type=int, default=200)
    parsers.add_argument('--eval_freq_iter', type=int, default=100)
    parsers.add_argument('--print_freq_iter', type=int, default=1)
    
    args = parsers.parse_args()
    dataloader_train, dataloader_test = get_dataloader(args=args)
    print(args)
    
    