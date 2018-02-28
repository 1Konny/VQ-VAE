import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

# utilize 1D latent feature spaces
class MODEL_MNIST(nn.Module):
    def __init__(self,k_dim=10,z_dim=64):
        super(MODEL_MNIST, self).__init__()
        self.z_dim = z_dim
        self.k_dim = k_dim

        # Encoder MLP
        self.encode = nn.Sequential(
            nn.Linear(784,1000),
            nn.ReLU(True),
            nn.Linear(1000,500),
            nn.ReLU(True),
            nn.Linear(500,300),
            nn.ReLU(True),
            nn.Linear(300,self.z_dim),
        )

        # Embedding Book
        self.embd = nn.Embedding(self.k_dim,self.z_dim).cuda()

        # Decoder MLP
        self.decode = nn.Sequential(
            nn.Linear(self.z_dim,300),
            nn.LeakyReLU(0.1,True),
            nn.Linear(300,500),
            nn.LeakyReLU(0.1,True),
            nn.Linear(500,1000),
            nn.LeakyReLU(0.1,True),
            nn.Linear(1000,784),
            nn.Tanh()
        )

    def find_nearest(self,query,target):
        Q=query.unsqueeze(1).repeat(1,target.size(0),1)
        T=target.unsqueeze(0).repeat(query.size(0),1,1)
        index=(Q-T).pow(2).sum(2).sqrt().min(1)[1]
        return target[index]

    def forward(self, X):
        Z_enc = self.encode(X.view(-1,784))
        Z_dec = self.find_nearest(Z_enc,self.embd.weight)
        Z_dec.register_hook(self.hook)

        X_recon = self.decode(Z_dec).view(-1,1,28,28)
        Z_enc_for_embd = self.find_nearest(self.embd.weight,Z_enc)
        return X_recon, Z_enc, Z_dec, Z_enc_for_embd

    def hook(self, grad):
        self.grad_for_encoder = grad
        return grad
