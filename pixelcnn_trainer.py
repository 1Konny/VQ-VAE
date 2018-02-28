
# coding: utf-8

# In[1]:


import numpy as np
import torch, torchvision, os, argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.autograd import Variable
from utils.data import data_loader
from utils.model_cifar10 import MODEL_CIFAR10
from utils.model_pixelcnn import PIXELCNN


# In[2]:


k_dim = 64
z_dim = 64
kernel_size = 5
fm = 128
batch_size = 32
max_epoch = 100
lr = 5e-4
data_dir = 'data'
train_dataset = 'CIFAR10'
test_dataset = 'CIFAR10_test'
ckpt_dir = os.path.join('checkpoints','cifar10_z64_k64')

train_data_args = argparse.Namespace(**{'batch_size':batch_size,'data_dir':data_dir,'dataset':train_dataset})
train_data, train_loader = data_loader(train_data_args)

test_data_args = argparse.Namespace(**{'batch_size':batch_size,'data_dir':data_dir,'dataset':test_dataset})
test_data, test_loader = data_loader(test_data_args)

model = MODEL_CIFAR10(k_dim=k_dim,z_dim=z_dim).cuda()
pixelcnn = PIXELCNN(k_dim=k_dim,z_dim=z_dim,kernel_size=kernel_size,fm=fm).cuda()
optimizer = optim.Adam(pixelcnn.parameters(),lr=lr,betas=(0.5,0.999))


# In[3]:


def load_checkpoint(ckpt_dir,model):
    filename = 'checkpoint.pth.tar'
    file_path = os.path.join(ckpt_dir,filename)
    if os.path.isfile(file_path):
        print("=> loading checkpoint '{}'".format(file_path))
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (iter {})"
              .format(filename, checkpoint['iter']))
    else:
        print("=> no checkpoint found at '{}'".format(file_path))


# In[4]:


load_checkpoint(ckpt_dir,model)
criterion = F.cross_entropy


# In[5]:


pixelcnn_dsets = dict()
pixelcnn_dsets['Z'] = []
pixelcnn_dsets['index'] = []
for batch_idx, (X,_) in enumerate(train_loader):
    X = Variable(X).cuda()
    Z_enc, embd_index = model.forward_pixel_cnn(X)
    pixelcnn_dsets['Z'].append(Z_enc.data)
    pixelcnn_dsets['index'].append(embd_index.view(Z_enc.size(0),Z_enc.size(2),Z_enc.size(3)).data)
    
pixelcnn_dsets['Z'] = torch.cat(pixelcnn_dsets['Z'],0)
pixelcnn_dsets['index'] = torch.cat(pixelcnn_dsets['index'],0)
print('Done!')


# In[ ]:


import torch.utils.data as data
from torch.utils.data import DataLoader

class pixel_dset(data.Dataset):
    def __init__(self, pixelcnn_dsets):
        self.data = pixelcnn_dsets['Z']
        self.labels = pixelcnn_dsets['index']

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.data)
    
new_loader = DataLoader(pixel_dset(pixelcnn_dsets),batch_size=batch_size)


# In[ ]:


pixelcnn.train()
for e in range(max_epoch):
    for batch_idx, (Z,index) in enumerate(new_loader):
        Z = Variable(Z).cuda()
        index = Variable(index).cuda()
        logits = pixelcnn(Z)
        loss = criterion(logits,index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(e,loss)


# In[ ]:


pixelcnn.eval()
# dummy_input = Variable(torch.rand(Z.size())).cuda()
dummy_input = Variable(save['Z'])
prior_logit = pixelcnn(dummy_input).permute(0,2,3,1)
prior = prior_logit.exp().div(1e-9 + prior_logit.exp().sum(3,True))

sample_index = torch.multinomial(prior.view(-1,64),1).squeeze()
sample_z = model._modules['embd'].weight[sample_index].view(100,8,8,64).permute(0,3,1,2)
sample = model.decode(sample_z)

grid=torchvision.utils.make_grid(sample.data,nrow=10,padding=5,pad_value=-1)
grid = (grid+1)/2
plt.imshow(grid.permute(1,2,0).cpu().numpy())
plt.show()

grid=torchvision.utils.make_grid(save['X'],nrow=10,padding=5,pad_value=-1)
grid = (grid+1)/2
plt.imshow(grid.permute(1,2,0).cpu().numpy())
plt.show()


# In[ ]:


model(X)[0]


# In[ ]:


pixelcnn.eval()
X = iter(test_loader).next()[0]
X = Variable(X.cuda())    


# In[ ]:


X[:,:,8:,0:]=0


# In[ ]:


Z_enc = model.forward_pixel_cnn(X)[0]
Z_enc[:,:,4:,0:]=0
prior_logit = pixelcnn(Z_enc).permute(0,2,3,1)
prior = prior_logit.exp().div(1e-9 + prior_logit.exp().sum(3,True))
sample_index = torch.multinomial(prior.view(-1,64),1).squeeze()
sample_z = model._modules['embd'].weight[sample_index].view(100,8,8,64).permute(0,3,1,2)
sample = model.decode(sample_z)


# In[ ]:


def show_batch_images(images,nrow=10,padding=5,pad_value=-1,**kwargs):
    grid=torchvision.utils.make_grid(images,nrow=10,padding=5,pad_value=-1,**kwargs)
    grid = (grid+1)/2
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.show()

show_batch_images(X.data)
show_batch_images(sample.data)
show_batch_images(model(X)[0].data)

