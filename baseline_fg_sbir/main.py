import os
import time
import torch
import argparse
import torch.utils.data as data 

from tqdm import tqdm
from dataset import FGSBIR_Dataset
from model import FGSBIR_Model
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['TORCH_CUDA_ARCH_LIST']="6.0"

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
    parsers.add_argument('--output_size', type=int, default=64)
    parsers.add_argument('--root_dir', type=str, default='./../')
    parsers.add_argument('--batch_size', type=int, default=16)
    parsers.add_argument('--step_size', type=int, default=100)
    parsers.add_argument('--gamma', type=float, default=0.5)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--learning_rate', type=float, default=0.001)
    parsers.add_argument('--epochs', type=int, default=200)
    parsers.add_argument('--eval_freq_iter', type=int, default=100)
    parsers.add_argument('--print_freq_iter', type=int, default=1)
    
    args = parsers.parse_args()
    dataloader_train, dataloader_test = get_dataloader(args=args)
    print(args)
    
    model = FGSBIR_Model(args=args)
    model.to(device)
    
    step_count, top1, top5, top10 = -1, 0, 0, 0
    
    scheduler = StepLR(model.optimizer, step_size=args.step_size, gamma=args.gamma)
    for i_epoch in range(args.epochs):
        print(f"Epoch: {i_epoch+1} / {args.epochs}")
        loss = 0
        for _, batch_data in enumerate(tqdm(dataloader_train)):
            step_count = step_count + 1
            start = time.time()
            model.train()
            loss = model.train_model(batch=batch_data)

        scheduler.step()
        with torch.no_grad():
            model.eval()
            top1_eval, top5_eval, top10_eval = model.evaluate(dataloader_test)
            
            if top1_eval > top1:
                top1, top10 = top1_eval, top10_eval
                torch.save(model.state_dict(), args.dataset_name + '_best.pth')
                torch.save({
                    'image_backbones': model.image_embedding_network.state_dict(),
                    'sketch_backbones': model.sketch_embedding_network.state_dict()
                }, f"{args.dataset_name}_backbones.pth")
                torch.save({
                    'positive_attention': model.positive_attention.state_dict(),
                    'negative_attention': model.negative_attention.state_dict(),
                    'sketch_attention': model.sketch_attention.state_dict()
                }, f"{args.dataset_name}_attentions.pth")
                
        
        # Load model
        # model = FGSBIR_Model(args)
        # model.load_state_dict(torch.load(f"{args.backbone_name}_{args.dataset_name}_best.pth"))

        # # Load backbone
        # model.sample_embedding_network.load_state_dict(torch.load(f"{args.backbone_name}_backbone.pth"))

        # # Load Linear layer
        # linear_state = torch.load(f"{args.backbone_name}_linears.pth")
        # model.positive_linear.load_state_dict(linear_state['positive_linear'])
        # model.negative_linear.load_state_dict(linear_state['negative_linear'])
        # model.sample_linear.load_state_dict(linear_state['sample_linear'])
        
        print('Top 1 accuracy:  {:.4f}'.format(top1_eval))
        print('Top 5 accuracy:  {:.4f}'.format(top5_eval))
        print('Top 10 accuracy: {:.4f}'.format(top10_eval))
        print('Loss:            {:.4f}'.format(loss))
        print("========================================")
