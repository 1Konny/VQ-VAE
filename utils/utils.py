import torch
import torch.nn as nn
from torch.autograd import Variable

class One_Hot(nn.Module):
    # got it from :
    # https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
    def __init__(self, depth):
        super(One_Hot,self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)
    def forward(self, X_in):
        X_in = X_in.long()
        return Variable(self.ones.index_select(0,X_in.data))
    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

def find_nearest(query, target):
    Q = query.unsqueeze(1).repeat(1,target.size(0),1)
    T = target.unsqueeze(0).repeat(query.size(0),1,1)
    index = (Q-T).pow(2).sum(2).sqrt().min(1)[1]

    return target[index]
