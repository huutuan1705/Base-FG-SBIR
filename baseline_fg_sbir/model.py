import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from tqdm import tqdm

from backbones import VGG16, ResNet50, InceptionV3
from attention import Attention_global, Linear_global

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Model(nn.Module):
    def __init__(self, args):
        super(FGSBIR_Model, self).__init__()
        self.sample_embedding_network = eval(args.backbone_name + "(args)")
        self.sketch_embedding_network = eval(args.backbone_name + "(args)")
        self.loss = nn.TripletMarginLoss(margin=args.margin)        
        self.sample_train_params = self.sample_embedding_network.parameters()
        self.sketch_train_params = self.sketch_embedding_network.parameters()
        self.args = args
        
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
        
        self.attention = Attention_global()
        self.attn_params = self.attention.parameters()
        
        self.sketch_attention = Attention_global()
        self.sketch_attn_params = self.sketch_attention.parameters()
        
        self.linear = Linear_global(feature_num=self.args.output_size)
        self.linear_params = self.linear.parameters()
        
        self.sketch_linear = Linear_global(feature_num=self.args.output_size)
        self.sketch_linear_params = self.sketch_linear.parameters()

        if self.args.use_kaiming_init:
            self.attention.apply(init_weights)
            self.sketch_attention.apply(init_weights)
            self.linear.apply(init_weights)
            self.sketch_linear.apply(init_weights)
            
        self.optimizer = optim.Adam([
            {'params': self.sketch_embedding_network.parameters(), 'lr': args.learning_rate},
            {'params': self.sample_embedding_network.parameters(), 'lr': args.learning_rate},
        ])
    
    def comput_loss(self, anchors, positive, negative):
        positive_expanded = positive.unsqueeze(1).expand(-1, self.args.num_anchors, -1)
        negative_expanded = negative.unsqueeze(1).expand(-1, self.args.num_anchors, -1)
        
        anchors_reshaped = anchors.reshape(-1, self.args.output_size)
        positive_reshaped = positive_expanded.reshape(-1, self.args.output_size)
        negative_reshaped = negative_expanded.reshape(-1, self.args.output_size)
        
        losses = self.loss(anchors_reshaped, positive_reshaped, negative_reshaped)
        
        return losses
        
    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
            
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        positive_feature = self.attention(positive_feature)
        positive_feature = self.linear(positive_feature)
        
        negative_feature = self.sample_embedding_network(batch['negative_img'].to(device))
        negative_feature = self.attention(negative_feature)
        negative_feature = self.linear(negative_feature)
        
        sketch_features = []
        sketch_imgs_tensor = batch['sketch_imgs'].to(device)
        for i in range(sketch_imgs_tensor.shape[0]):
            sketch_feature = self.sketch_embedding_network(sketch_imgs_tensor[i].to(device))
            sketch_feature = self.sketch_attention(sketch_feature)
            sketch_feature = self.sketch_linear(sketch_feature)
            sketch_features.append(sketch_feature)
        sketch_features = torch.stack(sketch_features, dim=0)
        
        loss = self.comput_loss(sketch_features, positive_feature, negative_feature)
        loss.backward()
        self.optimizer.step()

        return loss.item() 

    def test_forward(self, batch):
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        positive_feature = self.attention(positive_feature)
        positive_feature = self.linear(positive_feature)
        
        sketch_features = []
        sketch_imgs_tensor = batch['sketch_imgs'].to(device)
        for i in range(sketch_imgs_tensor.shape[0]):
            sketch_feature = self.sketch_embedding_network(sketch_imgs_tensor[i].to(device))
            sketch_feature = self.sketch_attention(sketch_feature)
            sketch_feature = self.sketch_linear(sketch_feature)
            sketch_features.append(sketch_feature)
        sketch_features = torch.stack(sketch_features, dim=0)
        
        # print("sketch_features shape: ", sketch_features.shape)  # (1, 25 64)
        # print("positive_feature shape: ", positive_feature.shape)# (1, 64)
        return sketch_features, positive_feature
    
    def evaluate(self, datloader_test):
        self.eval()
        
        sketch_array_tests = []
        sketch_names = []
        image_array_tests = []
        image_names = []
        
        for idx, batch in enumerate(tqdm(datloader_test)):
            sketch_features, positive_feature = self.test_forward(batch)
            sketch_array_tests.append(sketch_features)
            sketch_names.append(batch['sketch_path'])
            
            for i_num, positive_name in enumerate(batch['positive_path']):
                if positive_name not in image_names:
                    image_names.append(batch['positive_sample'][i_num])
                    image_array_tests.append(positive_feature[i_num])
                    
        sketch_array_tests = torch.stack(sketch_array_tests)
        image_array_tests = torch.stack(image_array_tests)
        
        print("sketch_array_tests shape: ", sketch_array_tests.shape)
        print("image_array_tests shape: ", image_array_tests.shape)
        
        num_steps = len(sketch_array_tests)
        
        avererage_area = []
        avererage_area_percentile = []
        
        rank_all = torch.zeros(len(sketch_array_tests), num_steps)
        rank_all_percentile = torch.zeros(len(sketch_array_tests), num_steps)
        
        for i_batch, sampled_batch in enumerate(sketch_array_tests):
            mean_rank = []
            mean_rank_percentile = []
            
            sketch_name = sketch_names[i_batch]
            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('')[:-1])
            position_query = image_names.index(sketch_query_name)
            
            sampled_batch.squeeze(0)
            for i_sketch in range(sampled_batch.shape[0]):
                target_distance = F.pairwise_distance(sampled_batch[i_sketch].unsqueeze(0).to(device), 
                                                      image_array_tests[position_query].unsqueeze(0).to(device))
                distance = F.pairwise_distance(sampled_batch[i_sketch].unsqueeze(0).to(device),
                                               image_array_tests.to(device))
                
                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()
                
                rank_all_percentile[i_batch, i_sketch] = (len(distance) - rank_all[i_batch, i_sketch]) / (len(distance) - 1)
                if rank_all[i_batch, i_sketch].item() == 0:
                    mean_rank.append(1.)
                else:
                    mean_rank.append(1/rank_all[i_batch, i_sketch].item())
                    mean_rank_percentile.append(rank_all_percentile[i_batch, i_sketch].item())
            
            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))

        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]
        
        meanMB = np.mean(avererage_area)
        meanMA = np.mean(avererage_area_percentile)
        
        return top1_accuracy, top5_accuracy, top10_accuracy, meanMA, meanMB