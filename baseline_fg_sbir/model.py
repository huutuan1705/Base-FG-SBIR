from torch.autograd import Variable
import torch.nn as nn
from backbones import InceptionV3
from torch import optim
import torch
from tqdm import tqdm
import torch.nn.functional as F
from attention import Attention_global, Linear_global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FGSBIR_Model(nn.Module):
    def __init__(self, args):
        super(FGSBIR_Model, self).__init__()
        self.sample_embedding_network = eval(args.backbone_name + '(args)')
        self.sketch_embedding_network = eval(args.backbone_name + '(args)')
        self.loss = nn.TripletMarginLoss(margin=args.margin)
        self.sample_train_params = self.sample_embedding_network.parameters()
        self.sketch_train_params = self.sketch_embedding_network.parameters()
        
        self.attention = Attention_global()
        self.linear = Linear_global(feature_num=64)
        
        self.sketch_attention = Attention_global()
        self.sketch_linear = Linear_global(feature_num=64)
        
        self.optimizer = optim.Adam([
            {'params': self.sample_train_params, 'lr': args.learning_rate},
            {'params': self.sketch_train_params, 'lr': args.learning_rate},
            # {'params': self.attention.parameters(), 'lr': args.learning_rate},
            # {'params': self.linear.parameters(), 'lr': args.learning_rate},
        ])
        self.args = args


    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()

        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        negative_feature = self.sample_embedding_network(batch['negative_img'].to(device))
        sketch_feature = self.sample_embedding_network(batch['sketch_img'].to(device))
        
        positive_feature = self.attention(positive_feature)
        negative_feature = self.attention(negative_feature)
        sketch_feature = self.attention(sketch_feature)
        
        # positive_feature = self.linear(self.attention(positive_feature))
        # negative_feature = self.linear(self.attention(negative_feature))
        # sketch_feature = self.sketch_linear(self.sketch_attention(sketch_feature))

        loss = self.loss(sketch_feature, positive_feature, negative_feature)
        loss.backward()
        self.optimizer.step()

        return loss.item() 

    def test_forward(self, batch):            #  this is being called only during evaluation
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        sketch_feature = self.sample_embedding_network(batch['sketch_img'].to(device))
        
        positive_feature = self.attention(positive_feature)
        sketch_feature = self.attention(sketch_feature)
        # positive_feature = self.linear(self.attention(positive_feature))
        # sketch_feature = self.sketch_linear(self.sketch_attention(sketch_feature))
        
        return sketch_feature.cpu(), positive_feature.cpu()
    
    def evaluate(self, datloader_Test):
        Image_Feature_ALL = []
        Image_Name = []
        Sketch_Feature_ALL = []
        Sketch_Name = []
   
        self.eval()
        for i_batch, sanpled_batch in enumerate(tqdm(datloader_Test)):
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

            # print(sketch_feature.unsqueeze(0).shape)
            # print(len(Image_Feature_ALL))
            distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
            target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                  Image_Feature_ALL[position_query].unsqueeze(0))
            # print("len distance: ", len(distance))
            # print("len target_distance: ", len(target_distance))
            rank[num] = distance.le(target_distance).sum()

        top1 = rank.le(1).sum().numpy() / rank.shape[0]
        top5 = rank.le(5).sum().numpy() / rank.shape[0]
        top10 = rank.le(10).sum().numpy() / rank.shape[0]

        
        return top1, top5, top10

    