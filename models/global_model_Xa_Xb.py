import torch.nn as nn
import os
import sys
sys.path.append(os.getcwd())

from models.mlp2 import MLP2
from models.vgg_like import VGGLIKE, VGGLIKE_CIFAR10

#####################################################################
class Party_A_Classification(nn.Module):
    def __init__(self, input_shape, args, dropout_rate=0.0):
        super().__init__()
        
        self.backbone = nn.ModuleList()

        if args.dataset == 'NUSWIDE':
            self.cross_model = MLP2(input_shape, dropout_rate)
            self.global_model_num_features = self.cross_model.layer2[0].out_features
            print('dropout_rate_Xa:{}'.format(dropout_rate))
        elif args.dataset == 'CIFAR10':
            self.cross_model = VGGLIKE_CIFAR10(input_shape)
            self.global_model_num_features = self.cross_model.features[10].out_channels * self.cross_model.avgpool.output_size[0] * self.cross_model.avgpool.output_size[1]
        else:
            self.cross_model = VGGLIKE(input_shape)
            self.global_model_num_features = self.cross_model.features[10].out_channels * self.cross_model.avgpool.output_size[0] * self.cross_model.avgpool.output_size[1]
        
        self.backbone.append(self.cross_model)
        # add classifier_head
        self.global_model_classifier_head_cat = nn.Linear(self.global_model_num_features*2, len(args.mul_classes))
        self.global_model_classifier_head_single = nn.Linear(self.global_model_num_features, len(args.mul_classes))

    def forward(self, input_X):
        x_cross = self.cross_model(input_X).flatten(start_dim=1)
        z = x_cross

        return z

#####################################################################
class Party_B_Classification(nn.Module):
    def __init__(self, input_shape, args, dropout_rate=0.0):
        super().__init__()
        
        self.backbone = nn.ModuleList()
    
        if args.dataset == 'NUSWIDE':
            self.cross_model = MLP2(input_shape, dropout_rate)
            print('dropout_rate_Xb:{}'.format(dropout_rate))
        elif args.dataset == 'CIFAR10':
            self.cross_model = VGGLIKE_CIFAR10(input_shape)
        else:
            self.cross_model = VGGLIKE(input_shape)

        self.backbone.append(self.cross_model)

    def forward(self, input_X):
        x_cross = self.cross_model(input_X).flatten(start_dim=1)
        z = x_cross

        return z


