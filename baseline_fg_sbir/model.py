import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.autograd import Variable
from backbones import InceptionV3, VGG16, ResNet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Model(nn.Module):
    def __init__(self, args):
        super(FGSBIR_Model, self).__init__()
        self.sample_embedding_network = eval(args.backbone_name + "_Network(args)")
        self.loss = nn.TripletMarginLoss(margin=0.2)
        self.sample_train_params = self.sample_embedding_network.parameters()
        self.optimizer = optim.Adam(self.sample_train_params, args.learning_rate)
        self.args = args
        
    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        
        
