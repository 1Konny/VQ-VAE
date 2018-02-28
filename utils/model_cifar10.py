import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

class res_block(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(res_block, self).__init__()
        self.conv1 = conv3x3(inplanes,planes,stride)
        self.conv2 = conv1x1(planes,planes,stride)
        self.relu = nn.ReLU(True)

        self.weight_init()

    def weight_init(self):
        for ms in self._modules:
            if isinstance(self._modules[ms],nn.Conv2d):
                nn.init.kaiming_normal(self._modules[ms].weight)
                try : self._modules[ms].bias.data.fill_(0)
                except : pass

    def forward(self, x):
        residual = x
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out

# utilize 2D latent feature spaces
class MODEL_CIFAR10(nn.Module):
    def __init__(self,k_dim=10,z_dim=64):
        super(MODEL_CIFAR10, self).__init__()
        self.k_dim = k_dim
        self.z_dim = z_dim
        h = z_dim

        # Build Encoder Blocks
        conv1 = nn.Conv2d(3,h,4,2,1)
        conv2 = nn.Conv2d(h,h,4,2,1)
        conv3 = res_block(h,h)
        conv4 = res_block(h,h)
        #conv5 = nn.Conv2d(h,z_dim,1,1,0)

        layers = []
        layers.append(conv1) # (B,3,3,32) -> (B,h,16,16)
        layers.append(conv2) # (B,h,16,16) -> (B,h,8,8)
        layers.append(res_block(h,h)) # (B,h,8,8) -> (B,h,8,8)
        layers.append(res_block(h,h)) # (B,h,8,8) -> (B,h,8,8)
        #layers.append(conv5) # (B,h,8,8) -> (B,z_dim,8,8)
        self.encode = nn.Sequential(*layers)

        # Embedding Book
        self.embd = nn.Embedding(self.k_dim,self.z_dim)

        # Build Decoder Blocks
        #conv1 = nn.Conv2d(z_dim,h,1,1,0)
        conv2 = res_block(h,h)
        conv3 = res_block(h,h)
        conv4 = nn.ConvTranspose2d(h,h,4,2,1)
        conv5 = nn.ConvTranspose2d(h,3,4,2,1)
        tanh = nn.Tanh()

        layers = []
        #layers.append(conv1) # (B,z_dim,8,8) -> (B,h,8,8)
        layers.append(conv2) # (B,h,8,8) -> (B,h,8,8)
        layers.append(conv3) # (B,h,8,8) -> (B,h,8,8)
        layers.append(conv4) # (B,h,8,8) -> (B,h,16,16)
        layers.append(conv5) # (B,h,16,16) -> (B,3,32,32)
        layers.append(tanh) # -> (B,3,32,32)

        self.decode = nn.Sequential(*layers)

        self.weight_init()

    def find_nearest(self,query,target):
        Q=query.unsqueeze(1).repeat(1,target.size(0),1)
        T=target.unsqueeze(0).repeat(query.size(0),1,1)
        index=(Q-T).pow(2).sum(2).sqrt().min(1)[1]
        return target[index]

    def hook(self, grad):
        self.grad_for_encoder = grad
        return grad

    def weight_init(self):
        for ms in self._modules:
            if ms == 'embd' :
                self._modules[ms].weight.data.uniform_()
                continue
            for m in self._modules[ms]:
                if isinstance(m,nn.Conv2d):
                    nn.init.kaiming_normal(m.weight)
                    try:m.bias.data.fill_(0)
                    except:pass

    def forward(self, x):
        Z_enc_ori = self.encode(x) # -> (B,C,W,H), C==z_dim
        z_bs, z_c, z_w, z_h = Z_enc_ori.size()

        Z_enc = Z_enc_ori.permute(0,2,3,1) # (B,C,W,H) -> (B,W,H,C)
        Z_enc = Z_enc.contiguous().view(-1,self.z_dim) # -> (B*W*H,C)

        Z_dec = self.find_nearest(Z_enc,self.embd.weight) # -> (B*W*H,C)
        Z_dec = Z_dec.view(z_bs,z_w,z_h,z_c) # -> (B,W,H,C)
        Z_dec = Z_dec.permute(0,3,1,2).contiguous() # (B,W,H,C) -> (B,C,W,H)
        Z_dec.register_hook(self.hook)

        X_recon = self.decode(Z_dec)
        Z_enc_for_embd = self.find_nearest(self.embd.weight,Z_enc) # -> (K,C)

        return X_recon, Z_enc_ori, Z_dec, Z_enc_for_embd
    
    def find_nearest_pixel_cnn(self,query,target):
        Q=query.unsqueeze(1).repeat(1,target.size(0),1)
        T=target.unsqueeze(0).repeat(query.size(0),1,1)
        index=(Q-T).pow(2).sum(2).sqrt().min(1)[1]
        return index
    
    def forward_pixel_cnn(self, x):
        Z_enc_ori = self.encode(x) # -> (B,C,W,H), C==z_dim
        z_bs, z_c, z_w, z_h = Z_enc_ori.size()

        Z_enc = Z_enc_ori.permute(0,2,3,1) # (B,C,W,H) -> (B,W,H,C)
        Z_enc = Z_enc.contiguous().view(-1,self.z_dim) # -> (B*W*H,C)

        index = self.find_nearest_pixel_cnn(Z_enc,self.embd.weight) # -> (B*W*H,C)
        return Z_enc_ori, index
