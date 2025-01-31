import time
import torch
import argparse
import torch.utils.data as data 

from dataset import FGSBIR_Dataset
from model import FGSBIR_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parsers.add_argument('--epochs', type=int, default=200)
    parsers.add_argument('--eval_freq_iter', type=int, default=100)
    parsers.add_argument('--print_freq_iter', type=int, default=1)
    
    args = parsers.parse_args()
    dataloader_train, dataloader_test = get_dataloader(args=args)
    print(args)
    
    model = FGSBIR_Model(args=args)
    model.to(device)
    
    step_count, top1, top10 = -1, 0, 0
    
    for i_epoch in range(args.epochs):
        for batch_data in dataloader_train:
            step_count = step_count + 1
            start = time.time()
            model.train()
            loss = model.train_model(batch=batch_data)

            if step_count % args.print_freq_iter == 0:
                print('Epoch: {}, Iteration: {}, Loss: {:.5f}, Top1_Accuracy: {:.5f}, Top10_Accuracy: {:.5f}, Time: {}'.format
                      (i_epoch, step_count, loss, top1, top10, time.time()-start))

            if step_count % args.eval_freq_iter == 0:
                with torch.no_grad():
                    top1_eval, top10_eval = model.evaluate(dataloader_test)
                    print('results : ', top1_eval, ' / ', top10_eval)

                if top1_eval > top1:
                    torch.save(model.state_dict(), args.backbone_name + '_' + args.dataset_name + '_model_best.pth')
                    top1, top10 = top1_eval, top10_eval
                    print('Model Updated')