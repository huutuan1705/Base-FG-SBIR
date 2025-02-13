import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from tqdm import tqdm

from backbones import VGG16, ResNet50, InceptionV3
from cbam import AttentionWithCBAM
from attention import AttentionImage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Model(nn.Module):
    def __init__(self, args):
        super(FGSBIR_Model, self).__init__()
        self.sketch_embedding_network = eval(args.backbone_name + "(args)")
        self.image_embedding_network = eval(args.backbone_name + "(args)")
        self.loss = nn.TripletMarginLoss(margin=args.margin)
        self.optimizer = optim.Adam([
            {'params': self.sketch_embedding_network.parameters(), 'lr': args.learning_rate},
            {'params': self.image_embedding_network.parameters(), 'lr': args.learning_rate},
        ])
        self.args = args
        
        # self.positive_attention = AttentionWithCBAM(in_channels=2048)
        # self.negative_attention = AttentionWithCBAM(in_channels=2048)
        # self.sketch_attention = AttentionWithCBAM(in_channels=2048)
        
        self.positive_attention = AttentionImage(input_size=2048)
        self.negative_attention = AttentionImage(input_size=2048)
        self.sketch_attention = AttentionImage(input_size=2048)
        
    
    def test_forward(self, batch):            #  this is being called only during evaluation
        sketch_feature = self.sketch_embedding_network(batch['sketch_img'].to(device))
        positive_feature = self.image_embedding_network(batch['positive_img'].to(device))
        
        positive_feature = self.positive_attention(positive_feature)
        sketch_feature = self.sketch_attention(sketch_feature)
        
        return sketch_feature, positive_feature
        
    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        
        positive_feature = self.image_embedding_network(batch['positive_img'].to(device))
        negative_feature = self.image_embedding_network(batch['negative_img'].to(device))
        sketch_feature = self.sketch_embedding_network(batch['sketch_img'].to(device))
        
        positive_feature = self.positive_attention(positive_feature)
        negative_feature = self.negative_attention(negative_feature)
        sketch_feature = self.sketch_attention(sketch_feature)

        loss = self.loss(sketch_feature, positive_feature, negative_feature)
        loss.backward()
        self.optimizer.step()

        return loss.item() 

    def evaluate(self, datloader_test):
        Image_Feature_ALL = []
        Image_Name = []
        Sketch_Feature_ALL = []
        Sketch_Name = []
        start_time = time.time()
        self.eval()
        for i_batch, sanpled_batch in enumerate(tqdm(datloader_test)):
            sketch_feature, positive_feature= self.test_forward(sanpled_batch)
            Sketch_Feature_ALL.extend(sketch_feature)
            Sketch_Name.extend(sanpled_batch['sketch_path'])

            for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
                if positive_name not in Image_Name:
                    Image_Name.append(sanpled_batch['positive_path'][i_num])
                    Image_Feature_ALL.append(positive_feature[i_num])

        rank = torch.zeros(len(Sketch_Name))
        Image_Feature_ALL = torch.stack(Image_Feature_ALL)

        for num, sketch_feature in enumerate(Sketch_Feature_ALL):
            s_name = Sketch_Name[num]
            sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
            position_query = Image_Name.index(sketch_query_name)

            distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
            target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                  Image_Feature_ALL[position_query].unsqueeze(0))

            rank[num] = distance.le(target_distance).sum()

        top1 = rank.le(1).sum().numpy() / rank.shape[0]
        top5 = rank.le(5).sum().numpy() / rank.shape[0]
        top10 = rank.le(10).sum().numpy() / rank.shape[0]

        # print('Time to EValuate:{}'.format(time.time() - start_time))
        return top1, top5, top10