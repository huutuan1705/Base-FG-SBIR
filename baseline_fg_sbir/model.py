import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from backbones import InceptionV3, VGG16, ResNet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

