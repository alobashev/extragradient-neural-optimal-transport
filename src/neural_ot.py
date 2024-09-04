import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class NeuralOptimalTransport(nn.Module):
    def __init__(self, DIM=3, H=128, ZD=2):
        super().__init__()

        # transport map
        self.T = nn.Sequential(
            nn.Linear(DIM+ZD, H),
            nn.ReLU(inplace=True), 
            nn.Linear(H, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, DIM)
        )

        # Kantorovich potential
        self.f = nn.Sequential(
            nn.Linear(DIM, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, 1)
        )

        self.T.apply(self.weights_init_mlp)
        self.T.apply(self.weights_init_mlp)

    @staticmethod
    def weights_init_mlp(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


    def count_params(self):
        print('T params:', np.sum([np.prod(p.shape) for p in self.T.parameters()]))
        print('f params:', np.sum([np.prod(p.shape) for p in self.f.parameters()]))


    def freeze(self, model):
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()    
        
    def unfreeze(self, model):
        for p in model.parameters():
            p.requires_grad_(True)
        model.train(True)
